// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

    void invalidate_eval_accumulators() {
        for (auto& a : acc_stack_) a.valid = false;
        for (auto& pending : pending_acc_stack_) pending.valid = false;
    }

    static std::int64_t now_ticks() { return std::chrono::steady_clock::now().time_since_epoch().count(); }
    static std::int64_t ticks_for_ms(long long ms) {
        return std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::milliseconds(ms)).count();
    }
    bool stop_requested_now() {
        if (shared_stop_flag_ != nullptr && shared_stop_flag_->load(std::memory_order_acquire)) {
            // Latch it into our own flag: every caller up the tree checks stop_requested_ after a child returns and
            // must unwind without storing the child's interrupted score.
            stop_requested_.store(true, std::memory_order_release);
            return true;
        }
        if (stop_requested_.load(std::memory_order_acquire)) return true;
        if (nodes_ >= node_limit_) {
            request_stop();
            return true;
        }
        if (hard_deadline_active_.load(std::memory_order_relaxed) && (nodes_ & 1023ULL) == 0ULL
            && now_ticks() >= hard_deadline_ticks_.load(std::memory_order_relaxed)) {
            request_stop();
            return true;
        }
        return false;
    }

    void add_embedding_row(const Q8Vector<std::int8_t>& table, int idx,
                           float sign, AccumulatorArray& out) const {
        if (idx < 0 || idx >= static_cast<int>(N_FEATURES)) return;
        const std::int8_t* row = table.data() + static_cast<std::size_t>(idx) * EMB_DIM;
        std::int16_t* outp = out.data();
#if defined(__AVX2__)
        const bool add = sign > 0.0f;
        std::size_t j = 0;
        for (; j + 32 <= EMB_DIM; j += 32) {
            const __m256i packed = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row + j));
            const __m128i lo8 = _mm256_castsi256_si128(packed);
            const __m128i hi8 = _mm256_extracti128_si256(packed, 1);
            const __m256i lo16 = _mm256_cvtepi8_epi16(lo8);
            const __m256i hi16 = _mm256_cvtepi8_epi16(hi8);
            __m256i out_lo = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(outp + j));
            __m256i out_hi = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(outp + j + 16));
            out_lo = add ? _mm256_add_epi16(out_lo, lo16)
                         : _mm256_sub_epi16(out_lo, lo16);
            out_hi = add ? _mm256_add_epi16(out_hi, hi16)
                         : _mm256_sub_epi16(out_hi, hi16);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(outp + j), out_lo);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(outp + j + 16), out_hi);
        }
        for (; j < EMB_DIM; ++j) {
            outp[j] = static_cast<std::int16_t>(
                outp[j] + (add ? static_cast<int>(row[j]) : -static_cast<int>(row[j])));
        }
#else
        for (std::size_t j = 0; j < EMB_DIM; ++j) {
            outp[j] = static_cast<std::int16_t>(
                outp[j] + (sign > 0.0f ? static_cast<int>(row[j])
                                       : -static_cast<int>(row[j])));
        }
#endif
    }

    void add_us_embedding_row(const NetworkWeights& w, int idx, float sign,
                              AccumulatorArray& out) const {
        add_embedding_row(us_embedding_q8(w), idx, sign, out);
    }

    // Optimization A: incrementally maintained minor+major piece count.
    // Updated in make_move/unmake_move (engine-level wrappers around the free
    // makemove/unmakemove). Initialized via seed_phase_from_board() on every
    // board reset. Lets phase classification be O(1) instead of O(64).
    static bool is_minor_major(char p) {
        return piece_info(p).minor_major != 0;
        return p == 'N' || p == 'B' || p == 'R' || p == 'Q'
            || p == 'n' || p == 'b' || p == 'r' || p == 'q';
    }

    static int count_minor_major(const Board& b) {
        int count = 0;
        for (int sq = 0; sq < 64; ++sq) if (is_minor_major(b.sq[sq])) ++count;
        return count;
    }

    static int phase_from_count(int count) {
        if (count <= 6) return PHASE_ENDGAME;
        if (count <= 10) return PHASE_MIDDLEGAME;
        return PHASE_OPENING;
    }

    // Phase used for weights / accumulator validity. When all phase slots alias one weights object (every AB20 / AB21
    // single-file model), the phase is irrelevant: pin it so accumulators stay valid across captures that cross
    // the old phase boundaries (the integer sums are identical either way).
    int eval_phase_for_count(int count) const {
        if (phase_weights_[0] && phase_weights_[0] == phase_weights_[1] && phase_weights_[1] == phase_weights_[2])
            return PHASE_OPENING;
        return phase_from_count(count);
    }

    // Active weights pointer ??must be set before any eval/accumulator op.
    inline const NetworkWeights* active_weights() const noexcept {
        if (current_phase_ < 0 || current_phase_ >= N_PHASES) return nullptr;
        const auto& sp = phase_weights_[current_phase_];
        return sp ? sp.get() : nullptr;
    }

    inline void sync_active_weights_pointer() noexcept {
    }

    // O(1) phase update ??uses the incrementally maintained count.
    void update_phase_for(const Board& /*b*/) {
        const int p = eval_phase_for_count(minor_major_count_);
        if (p != current_phase_) {
            current_phase_ = p;
            sync_active_weights_pointer();
        }
    }

    // Reseed minor_major_count_ from a fresh board; call after any non-incremental
    // board state change (parse_fen, set_position, new_game, child engine copy).
    void seed_phase_from_board() {
        minor_major_count_ = count_minor_major(board_);
        current_phase_ = -1;  // force re-derivation on next eval
        sync_active_weights_pointer();
    }

    // Engine-level wrappers around the free make/unmake functions. These keep
    // minor_major_count_ in sync incrementally so update_phase_for stays O(1).
    // (Calls to ::makemove / ::unmakemove use the global namespace explicitly
    // to avoid recursive name resolution back into these member methods.)
    void make_move(const Move& m, Undo& u) {
        ::makemove(board_, m, u);
        if (is_minor_major(u.captured)) --minor_major_count_;
        if (m.promo) ++minor_major_count_;
    }
    void unmake_move(const Undo& u) {
        // Reverse the deltas applied in make_move BEFORE undoing the board,
        // since u still reflects what was captured/promoted at make time.
        if (is_minor_major(u.captured)) ++minor_major_count_;
        if (u.m.promo) --minor_major_count_;
        ::unmakemove(board_, u);
    }

    // Tier-1: refresh via king cache. Replaces rebuild_acc_from_board on the hot
    // refresh path. Each perspective is keyed independently by its own king sq;
    // we diff the stored piece array against the current board and apply only
    // the deltas. Cache miss falls back to a full sum (same cost as scratch).
    inline std::size_t king_cache_index(int persp, int phase, int king_sq) const {
        return static_cast<std::size_t>(persp) * N_PHASES * 64
             + static_cast<std::size_t>(phase) * 64
             + static_cast<std::size_t>(king_sq);
    }

    // HalfKAv2_hm: apply one piece's contribution to a single-perspective
    // king cache entry. Kings ARE features (no skip). The orient flip and
    // bucket are derived from the king_sq passed in (the king sq this cache
    // entry is keyed by).
    void apply_perspective_piece_delta(KingCacheEntry& entry, const NetworkWeights& w,
                                       int persp, char piece, int sq, int king_sq, float sign) const {
        if (piece == '.') return;
        const int pt = halfka_piece_type(piece);
        if (pt < 0) return;
        const int pc_color = ANDYBOT_PIECE_IS_UPPER(piece) ? 0 : 1;

        const int o_ksq = orient_flip_sq(persp, king_sq, king_sq);
        const int bucket = HALFKA_KING_BUCKETS[o_ksq];
        if (bucket < 0) return;

        const int p_idx = pt * 2 + (pc_color != persp ? 1 : 0);
        const int o_sq  = orient_flip_sq(persp, sq, king_sq);
        const int idx   = o_sq + p_idx * 64 + bucket * HALFKA_NUM_PLANES;
        if (idx < 0 || idx >= static_cast<int>(N_FEATURES)) return;

        add_us_embedding_row(w, idx, sign, entry.us);
    }

    // Tier-1 step 3 attempted: register-tiled refresh.
    // EXPERIMENTAL RESULT: net regression (~-19% aggregate). Same failure mode
    // as fused multi-op apply ??the FMA chain through tile accumulator
    // registers stalls the OOO pipeline more than the memory savings buy.
    // Reverted to the original per-delta multi-pass which the compiler
    // already optimizes into tight, dependency-free streaming SIMD. Save the
    // tiled approach for a threat-input network (HalfKA+Threats has 5-10x more active features
    // per refresh, where chain amortization actually works).

    // Feature index of `piece` on `sq` for one perspective whose king is on
    // king_sq; -1 for empty squares/invalid. Mirrors apply_perspective_piece_delta.
    int perspective_feature_index(int persp, char piece, int sq, int king_sq) const {
        if (piece == '.') return -1;
        const int pt = halfka_piece_type(piece);
        if (pt < 0) return -1;
        const int pc_color = ANDYBOT_PIECE_IS_UPPER(piece) ? 0 : 1;
        const int bucket = HALFKA_KING_BUCKETS[orient_flip_sq(persp, king_sq, king_sq)];
        if (bucket < 0) return -1;
        const int p_idx = pt * 2 + (pc_color != persp ? 1 : 0);
        const int idx = orient_flip_sq(persp, sq, king_sq) + p_idx * 64
                      + bucket * HALFKA_NUM_PLANES;
        return (idx >= 0 && idx < static_cast<int>(N_FEATURES)) ? idx : -1;
    }

    // Register-tiled king-cache update: entry += sum(rows[adds]) - sum(rows[subs]),
    // written to both the cache entry and the output accumulator. One pass over
    // the accumulator instead of one read-modify-write pass per delta, and no
    // separate copy. int16 wrapping add/sub is order-independent, so the result
    // is bit-identical to the per-delta path. (The earlier tiled attempt
    // regressed on fp32 FMA chains; these are 1-cycle integer ops.)
    static void tiled_refresh_q8(const std::int8_t* table,
                                 const int* adds, int n_add, const int* subs, int n_sub,
                                 AccumulatorValue* entry, AccumulatorValue* out) {
#if defined(__AVX512BW__)
        // 8 zmm x 32 lanes = 256 int16 per tile, 4 tiles.
        for (std::size_t base = 0; base < EMB_DIM; base += 256) {
            __m512i a0 = _mm512_loadu_si512(entry + base + 0),   a1 = _mm512_loadu_si512(entry + base + 32);
            __m512i a2 = _mm512_loadu_si512(entry + base + 64),  a3 = _mm512_loadu_si512(entry + base + 96);
            __m512i a4 = _mm512_loadu_si512(entry + base + 128), a5 = _mm512_loadu_si512(entry + base + 160);
            __m512i a6 = _mm512_loadu_si512(entry + base + 192), a7 = _mm512_loadu_si512(entry + base + 224);
#define ANDYBOT_TILE_ROW(OP, IDX)                                                              \
            {                                                                                  \
                const std::int8_t* r = table + static_cast<std::size_t>(IDX) * EMB_DIM + base; \
                const auto w8 = [&](int k) { return _mm512_cvtepi8_epi16(                       \
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + k))); };            \
                a0 = OP(a0, w8(0));   a1 = OP(a1, w8(32));  a2 = OP(a2, w8(64));  a3 = OP(a3, w8(96)); \
                a4 = OP(a4, w8(128)); a5 = OP(a5, w8(160)); a6 = OP(a6, w8(192)); a7 = OP(a7, w8(224)); \
            }
            for (int i = 0; i < n_sub; ++i) ANDYBOT_TILE_ROW(_mm512_sub_epi16, subs[i])
            for (int i = 0; i < n_add; ++i) ANDYBOT_TILE_ROW(_mm512_add_epi16, adds[i])
#undef ANDYBOT_TILE_ROW
            const __m512i v[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
            for (int k = 0; k < 8; ++k) {
                _mm512_storeu_si512(entry + base + 32 * k, v[k]);
                _mm512_storeu_si512(out + base + 32 * k, v[k]);
            }
        }
#elif defined(__AVX2__)
        // 8 ymm x 16 lanes = 128 int16 per tile, 8 tiles.
        for (std::size_t base = 0; base < EMB_DIM; base += 128) {
            const auto ld = [&](std::size_t k) {
                return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(entry + base + k)); };
            __m256i a0 = ld(0), a1 = ld(16), a2 = ld(32), a3 = ld(48);
            __m256i a4 = ld(64), a5 = ld(80), a6 = ld(96), a7 = ld(112);
#define ANDYBOT_TILE_ROW(OP, IDX)                                                              \
            {                                                                                  \
                const std::int8_t* r = table + static_cast<std::size_t>(IDX) * EMB_DIM + base; \
                const auto w8 = [&](int k) { return _mm256_cvtepi8_epi16(                       \
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(r + k))); };               \
                a0 = OP(a0, w8(0));  a1 = OP(a1, w8(16)); a2 = OP(a2, w8(32));  a3 = OP(a3, w8(48)); \
                a4 = OP(a4, w8(64)); a5 = OP(a5, w8(80)); a6 = OP(a6, w8(96));  a7 = OP(a7, w8(112)); \
            }
            for (int i = 0; i < n_sub; ++i) ANDYBOT_TILE_ROW(_mm256_sub_epi16, subs[i])
            for (int i = 0; i < n_add; ++i) ANDYBOT_TILE_ROW(_mm256_add_epi16, adds[i])
#undef ANDYBOT_TILE_ROW
            const __m256i v[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
            for (int k = 0; k < 8; ++k) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(entry + base + 16 * k), v[k]);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + base + 16 * k), v[k]);
            }
        }
#else
        for (std::size_t j = 0; j < EMB_DIM; ++j) {
            int value = entry[j];
            for (int i = 0; i < n_sub; ++i) value -= table[static_cast<std::size_t>(subs[i]) * EMB_DIM + j];
            for (int i = 0; i < n_add; ++i) value += table[static_cast<std::size_t>(adds[i]) * EMB_DIM + j];
            entry[j] = out[j] = static_cast<AccumulatorValue>(value);
        }
#endif
    }

    void refresh_perspective_via_cache(const NetworkWeights& w, const Board& b,
                                       int persp, int phase, int king_sq,
                                       AccumulatorArray& out_us
                                       ) {
        KingCacheEntry& entry = king_cache_[king_cache_index(persp, phase, king_sq)];

        if (!entry.valid) {
            entry.us.fill(AccumulatorValue{});
            for (int sq = 0; sq < 64; ++sq) {
                const char p = b.sq[sq];
                apply_perspective_piece_delta(entry, w, persp, p, sq, king_sq, +1.0f);
                entry.pieces[sq] = p;
            }
            std::memcpy(entry.piece_bbs, b.pieces, sizeof(entry.piece_bbs));
            entry.valid = true;
        } else {
            int tiled_adds[64], tiled_subs[64];   // at most one of each per square
            int n_tiled_adds = 0, n_tiled_subs = 0;
            bb::Bitboard changed = 0;
            for (int color = 0; color < 2; ++color)
                for (int kind = 0; kind < 6; ++kind)
                    changed |= entry.piece_bbs[color][kind] ^ b.pieces[color][kind];
            while (changed != 0) {
                const int sq = bb::lsb(changed);
                changed &= changed - 1;
                const char old_p = entry.pieces[sq];
                const char new_p = b.sq[sq];
                if (old_p == new_p) continue;
                // Collect deltas; applied in one tiled pass below.
                const int old_idx = perspective_feature_index(persp, old_p, sq, king_sq);
                const int new_idx = perspective_feature_index(persp, new_p, sq, king_sq);
                if (old_idx >= 0) tiled_subs[n_tiled_subs++] = old_idx;
                if (new_idx >= 0) tiled_adds[n_tiled_adds++] = new_idx;
                entry.pieces[sq] = new_p;
            }
            std::memcpy(entry.piece_bbs, b.pieces, sizeof(entry.piece_bbs));
            // Writes the updated entry and the output in the same pass, so the
            // separate copy below is skipped on this path.
            tiled_refresh_q8(us_embedding_q8(w).data(), tiled_adds, n_tiled_adds,
                             tiled_subs, n_tiled_subs, entry.us.data(), out_us.data());
            return;
        }

        out_us = entry.us;
    }

    void refresh_acc_via_king_cache(const Board& b, EvalAccumulator& out) {
        out.valid = false;
        if (!onb(b.king[WHITE]) || !onb(b.king[BLACK])) return;
        out.white_king_sq = b.king[WHITE];
        out.black_king_sq = b.king[BLACK];
        out.phase = current_phase_;
        // precompute buckets so subsequent apply_piece_delta calls during
        // make_move stay O(1) per piece.
        const int o_wk = orient_flip_sq(0, out.white_king_sq, out.white_king_sq);
        const int o_bk = orient_flip_sq(1, out.black_king_sq, out.black_king_sq);
        out.white_bucket = HALFKA_KING_BUCKETS[o_wk];
        out.black_bucket = HALFKA_KING_BUCKETS[o_bk];
        if (out.white_bucket < 0 || out.black_bucket < 0) return;

        const NetworkWeights* wp = active_weights();
        if (!wp) return;
        const NetworkWeights& w = *wp;
        const int phase = current_phase_;
        if (phase < 0 || phase >= N_PHASES) return;

        refresh_perspective_via_cache(w, b, 0, phase, b.king[WHITE], out.us_white
                                      );
        refresh_perspective_via_cache(w, b, 1, phase, b.king[BLACK], out.us_black
                                      );

        out.valid = true;
    }

    // Tier-1 step 2 attempted: fused multi-op delta apply (templated by N).
    // EXPERIMENTAL RESULT: net regression for HalfKP at EMB_DIM=768. The
    // build_child path only fuses 2-3 deltas per move; the dependency chain
    // through `acc` (n FMAs in series per j-step) costs more than the
    // memory-pass savings vs the original `out = parent; for each delta:
    // add_embedding_row(out)` pattern (which has zero cross-j dependencies
    // and so saturates OOO bandwidth). Reverted. The fused path is still
    // worth trying for HalfKA+Threats (~32 + ~128 = ~160 active features) in
    // a threat-input network, where fusing is over many more deltas.

    int accumulator_feature_index(const EvalAccumulator& acc, int persp,
                                  char piece, int sq) const {
            if (piece == '.') return -1;
            const int pt = halfka_piece_type(piece);
            if (pt < 0) return -1;
            const int pc_color = ANDYBOT_PIECE_IS_UPPER(piece) ? 0 : 1;
            const int king_sq = persp == WHITE ? acc.white_king_sq : acc.black_king_sq;
            const int bucket = persp == WHITE ? acc.white_bucket : acc.black_bucket;
            const int p_idx = pt * 2 + (pc_color != persp ? 1 : 0);
            const int o_sq = orient_flip_sq(persp, sq, king_sq);
            const int idx = o_sq + p_idx * 64 + bucket * HALFKA_NUM_PLANES;
            return idx >= 0 && idx < static_cast<int>(N_FEATURES) ? idx : -1;
    }

    void fuse_three_accumulator_deltas(const AccumulatorArray& src,
                                       AccumulatorArray& dst,
                                       const Q8Vector<std::int8_t>& table,
                                       const std::array<int, 3>& indices,
                                       const std::array<int, 3>& signs) const {
            std::size_t j = 0;
#if defined(__AVX2__)
            for (; j + 16 <= EMB_DIM; j += 16) {
                __m256i value = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(src.data() + j));
                for (int d = 0; d < 3; ++d) {
                    if (indices[d] < 0) continue;
                    const std::int8_t* row = table.data()
                        + static_cast<std::size_t>(indices[d]) * EMB_DIM + j;
                    const __m256i delta = _mm256_cvtepi8_epi16(
                        _mm_loadu_si128(reinterpret_cast<const __m128i*>(row)));
                    value = signs[d] > 0 ? _mm256_add_epi16(value, delta)
                                         : _mm256_sub_epi16(value, delta);
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data() + j), value);
            }
#endif
            for (; j < EMB_DIM; ++j) {
                std::int16_t value = src[j];
                for (int d = 0; d < 3; ++d) {
                    if (indices[d] < 0) continue;
                    const int delta = table[static_cast<std::size_t>(indices[d]) * EMB_DIM + j];
                    value = static_cast<std::int16_t>(value + (signs[d] > 0 ? delta : -delta));
                }
                dst[j] = value;
            }
    }

    void fuse_move_accumulator_deltas(const AccumulatorArray& src,
                                      AccumulatorArray& dst,
                                      const Q8Vector<std::int8_t>& table,
                                      int removed_index, int added_index,
                                      int captured_index) const {
        const std::int8_t* removed = table.data()
            + static_cast<std::size_t>(removed_index) * EMB_DIM;
        const std::int8_t* added = table.data()
            + static_cast<std::size_t>(added_index) * EMB_DIM;
        const std::int8_t* captured = captured_index >= 0
            ? table.data() + static_cast<std::size_t>(captured_index) * EMB_DIM
            : nullptr;
        std::size_t j = 0;
#if ANDYBOT_AVX512 && (defined(__AVX512BW__) || defined(_M_AVX512BW))
        if (captured != nullptr) {
            for (; j + 32 <= EMB_DIM; j += 32) {
                __m512i value = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(src.data() + j));
                value = _mm512_sub_epi16(value, _mm512_cvtepi8_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(removed + j))));
                value = _mm512_add_epi16(value, _mm512_cvtepi8_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(added + j))));
                value = _mm512_sub_epi16(value, _mm512_cvtepi8_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(captured + j))));
                _mm512_storeu_si512(reinterpret_cast<void*>(dst.data() + j), value);
            }
        } else {
            for (; j + 32 <= EMB_DIM; j += 32) {
                __m512i value = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(src.data() + j));
                value = _mm512_sub_epi16(value, _mm512_cvtepi8_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(removed + j))));
                value = _mm512_add_epi16(value, _mm512_cvtepi8_epi16(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(added + j))));
                _mm512_storeu_si512(reinterpret_cast<void*>(dst.data() + j), value);
            }
        }
#elif defined(__AVX2__)
        if (captured != nullptr) {
            for (; j + 16 <= EMB_DIM; j += 16) {
                __m256i value = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(src.data() + j));
                value = _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(removed + j))));
                value = _mm256_add_epi16(value, _mm256_cvtepi8_epi16(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(added + j))));
                value = _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(captured + j))));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data() + j), value);
            }
        } else {
            for (; j + 16 <= EMB_DIM; j += 16) {
                __m256i value = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(src.data() + j));
                value = _mm256_sub_epi16(value, _mm256_cvtepi8_epi16(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(removed + j))));
                value = _mm256_add_epi16(value, _mm256_cvtepi8_epi16(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(added + j))));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data() + j), value);
            }
        }
#endif
        for (; j < EMB_DIM; ++j) {
            int value = static_cast<int>(src[j])
                - static_cast<int>(removed[j]) + static_cast<int>(added[j]);
            if (captured != nullptr) value -= static_cast<int>(captured[j]);
            dst[j] = static_cast<std::int16_t>(value);
        }
    }

    bool build_child_acc_from_parent(const EvalAccumulator& parent, const Board& child, const Move& m, const Undo& u, EvalAccumulator& out) {
        if (!parent.valid) return false;
        if (parent.phase != current_phase_) return false;
        if (child.king[WHITE] != u.kw || child.king[BLACK] != u.kb) return false;
        const NetworkWeights* wp = active_weights();
        if (!wp) return false;
        const NetworkWeights& w = *wp;
        out.valid = true;
        out.phase = parent.phase;
        out.white_king_sq = parent.white_king_sq;
        out.black_king_sq = parent.black_king_sq;
        out.white_bucket = parent.white_bucket;
        out.black_bucket = parent.black_bucket;

        int cap_sq = m.to;
        if (m.ep) cap_sq = isw(u.moved) ? (m.to - 8) : (m.to + 8);
        const std::array<int, 3> signs{-1, +1, -1};
        const std::array<int, 3> white_indices{
            accumulator_feature_index(parent, WHITE, u.moved, m.from),
            accumulator_feature_index(parent, WHITE, child.sq[m.to], m.to),
            accumulator_feature_index(parent, WHITE, u.captured, cap_sq)};
        const std::array<int, 3> black_indices{
            accumulator_feature_index(parent, BLACK, u.moved, m.from),
            accumulator_feature_index(parent, BLACK, child.sq[m.to], m.to),
            accumulator_feature_index(parent, BLACK, u.captured, cap_sq)};
        fuse_move_accumulator_deltas(parent.us_white, out.us_white, us_embedding_q8(w),
                                     white_indices[0], white_indices[1], white_indices[2]);
        fuse_move_accumulator_deltas(parent.us_black, out.us_black, us_embedding_q8(w),
                                     black_indices[0], black_indices[1], black_indices[2]);
        return true;
    }

    bool build_king_child_partially(const EvalAccumulator& parent, const Board& child,
                                    const Move& m, const Undo& u,
                                    EvalAccumulator& out) {
        if (!parent.valid || m.castle) return false;
        if (parent.phase != current_phase_) return false;
        // A capture can cross a model-phase boundary before eval_cp() updates
        // current_phase_. In that case use the ordinary full refresh later.
        if (eval_phase_for_count(minor_major_count_) != current_phase_) return false;
        const int moving = isw(u.moved) ? WHITE : BLACK;
        const int stationary = moving ^ 1;
        if (child.king[moving] == (moving == WHITE ? u.kw : u.kb)
            || child.king[stationary] != (stationary == WHITE ? u.kw : u.kb)) {
            return false;
        }
        const NetworkWeights* wp = active_weights();
        if (!wp) return false;
        const NetworkWeights& w = *wp;

        out.valid = false;
        out.phase = parent.phase;
        out.white_king_sq = child.king[WHITE];
        out.black_king_sq = child.king[BLACK];
        out.white_bucket = HALFKA_KING_BUCKETS[
            orient_flip_sq(WHITE, out.white_king_sq, out.white_king_sq)];
        out.black_bucket = HALFKA_KING_BUCKETS[
            orient_flip_sq(BLACK, out.black_king_sq, out.black_king_sq)];
        if (out.white_bucket < 0 || out.black_bucket < 0) return false;

        if (moving == WHITE) {
            refresh_perspective_via_cache(w, child, WHITE, current_phase_,
                                          child.king[WHITE], out.us_white
                                          );
        } else {
            refresh_perspective_via_cache(w, child, BLACK, current_phase_,
                                          child.king[BLACK], out.us_black
                                          );
        }

        int cap_sq = m.to;
        if (m.ep) cap_sq = isw(u.moved) ? (m.to - 8) : (m.to + 8);
        const std::array<int, 3> signs{-1, +1, -1};
        const std::array<int, 3> indices{
            accumulator_feature_index(parent, stationary, u.moved, m.from),
            accumulator_feature_index(parent, stationary, child.sq[m.to], m.to),
            accumulator_feature_index(parent, stationary, u.captured, cap_sq)};
        if (stationary == WHITE) {
            fuse_three_accumulator_deltas(parent.us_white, out.us_white,
                                          us_embedding_q8(w), indices, signs);
        } else {
            fuse_three_accumulator_deltas(parent.us_black, out.us_black,
                                          us_embedding_q8(w), indices, signs);
        }
        out.valid = true;
        return true;
    }

    void prepare_child_accumulator(int ply, const Move& m, const Undo& u) {
        if (!model_loaded_) return;
        const int pidx = std::min(ply, MAX_PLY);
        const int cidx = std::min(ply + 1, MAX_PLY + 1);
        const int parent_acc_idx =
            (pending_acc_stack_[pidx].valid && pending_acc_stack_[pidx].null_move)
                ? pending_acc_stack_[pidx].parent_acc_idx
                : pidx;
        // King moves refresh the king cache as an observable side effect.
        // Deferring and then skipping that refresh on a TT cutoff changes the
        // cache's later floating-point delta order. Keep king moves eager;
        // only ordinary copy-and-delta children are safe to defer.
        // Exact: accumulators are integers now, so the order in which king-cache entries are refreshed no longer
        // affects any value. A king move is deferred like every other move and built on first use; a child that
        // returns from the table or is never evaluated costs nothing. (Bare endgames are mostly king moves.)
        pending_acc_stack_[cidx].king_move = (u.moved == 'K' || u.moved == 'k');
        if (false) {
            pending_acc_stack_[cidx].valid = false;
            if (!build_king_child_partially(
                    acc_stack_[parent_acc_idx], board_, m, u, acc_stack_[cidx])) {
                refresh_acc_via_king_cache(board_, acc_stack_[cidx]);
            }
            return;
        }
        acc_stack_[cidx].valid = false;
        pending_acc_stack_[cidx].valid = true;
        pending_acc_stack_[cidx].null_move = false;
        pending_acc_stack_[cidx].parent_acc_idx = parent_acc_idx;
        pending_acc_stack_[cidx].move = m;
        pending_acc_stack_[cidx].undo = u;
    }

    void prepare_null_accumulator(int ply) {
        if (!model_loaded_) return;
        const int pidx = std::min(ply, MAX_PLY);
        const int cidx = std::min(ply + 1, MAX_PLY + 1);
        acc_stack_[cidx].valid = false;
        pending_acc_stack_[cidx].valid = true;
        pending_acc_stack_[cidx].null_move = true;
        pending_acc_stack_[cidx].parent_acc_idx =
            (pending_acc_stack_[pidx].valid && pending_acc_stack_[pidx].null_move)
                ? pending_acc_stack_[pidx].parent_acc_idx
                : pidx;
    }

    void materialize_accumulator(int ply) {
        if (!model_loaded_) return;
        const int idx = std::min(ply, MAX_PLY + 1);
        PendingAccumulator& pending = pending_acc_stack_[idx];
        if (!pending.valid) {
            return;
        }
        const int pidx =
            pending.parent_acc_idx;
        if (pending.null_move) {
            acc_stack_[idx] = acc_stack_[pidx];
        } else if (pending.king_move) {
            if (!build_king_child_partially(acc_stack_[pidx], board_, pending.move, pending.undo, acc_stack_[idx]))
                refresh_acc_via_king_cache(board_, acc_stack_[idx]);
        } else if (!build_child_acc_from_parent(
                       acc_stack_[pidx], board_, pending.move, pending.undo, acc_stack_[idx])) {
            refresh_acc_via_king_cache(board_, acc_stack_[idx]);
        }
        pending.valid = false;
    }

    int eval_from_accumulator(const EvalAccumulator& acc, int stm) const {
        const NetworkWeights* wp = active_weights();
        if (!wp) return 0;
        const NetworkWeights& w = *wp;
        // One small head per material bucket, selected by piece count (both kings included).
        const std::size_t head_index = static_cast<std::size_t>((bb::popcount(board_.occ) - 1) / 4) & 7u;
        const auto& hd = w.output_heads[head_index];
        const AccumulatorArray& us_src =
            (stm == WHITE) ? acc.us_white : acc.us_black;
        const AccumulatorArray& them_src =
            (stm == WHITE) ? acc.us_black : acc.us_white;

        std::array<float, L1_OUT> h2;
        std::array<std::uint8_t, L1_IN> x_q8;
        // Indices of the non-zero groups of four projection inputs (+8 slack for the last vector store).
        alignas(64) std::array<std::uint32_t, L1_IN / 4 + 8> nnz_index;
        int nnz_count = -1;
        // Integer-native pairwise activation. Accumulator channels use their
        // feature-table scales; the precomputed Q15 factor maps each product
        // directly to the grouped projection's [0,127] activation domain.
        {
        bool pairs_done = false;
#if defined(__AVX512BW__)
        if (w.pair_uniform) {
            // Exact int16 form of the generic Q15 path (see set_pair_uniform): 32 channels per step.
            const __m512i lo = _mm512_setzero_si512();
            const __m512i hi = _mm512_set1_epi16(w.pair_clip);
            const __m512i mult = _mm512_set1_epi16(w.pair_mult);
            const __m512i max_u7 = _mm512_set1_epi16(127);
            const auto make_pairs16 = [&](const AccumulatorArray& source, std::uint8_t* destination) {
                for (std::size_t j = 0; j < PAIRWISE_HALF; j += 32) {
                    const __m512i a = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_loadu_si512(reinterpret_cast<const void*>(source.data() + j)), lo), hi);
                    const __m512i b = _mm512_min_epi16(_mm512_max_epi16(
                        _mm512_loadu_si512(reinterpret_cast<const void*>(source.data() + j + PAIRWISE_HALF)), lo), hi);
                    const __m512i q = _mm512_min_epi16(
                        _mm512_mulhrs_epi16(_mm512_mullo_epi16(a, b), mult), max_u7);
                    const __m256i bytes = _mm512_cvtepi16_epi8(q);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + j), bytes);
#if ANDYBOT_AVX512
                    // Record which groups of four bytes are non-zero, as group indices.
                    const __mmask8 nz = _mm256_test_epi32_mask(bytes, bytes);
                    const int g0 = static_cast<int>((destination - x_q8.data() + static_cast<std::ptrdiff_t>(j)) / 4);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(nnz_index.data() + nnz_count),
                        _mm256_maskz_compress_epi32(nz, _mm256_add_epi32(_mm256_set1_epi32(g0),
                                                                         _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7))));
                    nnz_count += static_cast<int>(_mm_popcnt_u32(nz));
#endif
                }
            };
#if ANDYBOT_AVX512
            nnz_count = 0;
#endif
            make_pairs16(us_src, x_q8.data());
            make_pairs16(them_src, x_q8.data() + PAIRWISE_HALF);
            pairs_done = true;
        }
#elif !ANDYBOT_AVX512 && defined(__AVX2__)
        if (w.pair_uniform) {
            // AVX2 form of the exact int16 pairwise path above (same ops per lane): 32 channels per step.
            const __m256i lo = _mm256_setzero_si256();
            const __m256i hi = _mm256_set1_epi16(w.pair_clip);
            const __m256i mult = _mm256_set1_epi16(w.pair_mult);
            const __m256i max_u7 = _mm256_set1_epi16(127);
            const auto make_pairs16 = [&](const AccumulatorArray& source, std::uint8_t* destination) {
                const auto q16 = [&](std::size_t j) {
                    const __m256i a = _mm256_min_epi16(_mm256_max_epi16(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source.data() + j)), lo), hi);
                    const __m256i b = _mm256_min_epi16(_mm256_max_epi16(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source.data() + j + PAIRWISE_HALF)), lo), hi);
                    return _mm256_min_epi16(_mm256_mulhrs_epi16(_mm256_mullo_epi16(a, b), mult), max_u7);
                };
                for (std::size_t j = 0; j < PAIRWISE_HALF; j += 32) {
                    // values are in [0,127] so packus is exact; permute restores order after the per-lane pack.
                    const __m256i packed = _mm256_permute4x64_epi64(
                        _mm256_packus_epi16(q16(j), q16(j + 16)), 0xD8);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + j), packed);
#if !ANDYBOT_AVX512
                    // Non-zero groups of four bytes -> group indices via a 256-entry table (no branches).
                    const unsigned nz = ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                        _mm256_cmpeq_epi32(packed, lo)))) & 0xFFu;
                    const int g0 = static_cast<int>((destination - x_q8.data() + static_cast<std::ptrdiff_t>(j)) / 4);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(nnz_index.data() + nnz_count),
                        _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(NNZ_LUT8[nz].data())),
                                         _mm256_set1_epi32(g0)));
                    nnz_count += _mm_popcnt_u32(nz);
#endif
                }
            };
#if !ANDYBOT_AVX512
            nnz_count = 0;
#endif
            make_pairs16(us_src, x_q8.data());
            make_pairs16(them_src, x_q8.data() + PAIRWISE_HALF);
            pairs_done = true;
        }
#endif
        if (!pairs_done) {
#if ANDYBOT_AVX512 && (defined(__AVX512F__) || defined(_M_AVX512F))
        const __m512i zero_i32 = _mm512_setzero_si512();
        const __m512i max_u7 = _mm512_set1_epi32(127);
        const __m512i rounding_q15 = _mm512_set1_epi32(1 << 14);
        const auto make_pairs = [&](const AccumulatorArray& source,
                                    const std::array<std::int16_t, EMB_DIM>& clip,
                                    const std::array<std::int32_t, PAIRWISE_HALF>& multiplier,
                                    std::uint8_t* destination) {
            for (std::size_t j = 0; j < PAIRWISE_HALF; j += 16) {
                __m512i a = _mm512_cvtepi16_epi32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(source.data() + j)));
                __m512i b = _mm512_cvtepi16_epi32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(source.data() + j + PAIRWISE_HALF)));
                const __m512i max_a = _mm512_cvtepi16_epi32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(clip.data() + j)));
                const __m512i max_b = _mm512_cvtepi16_epi32(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(clip.data() + j + PAIRWISE_HALF)));
                a = _mm512_min_epi32(_mm512_max_epi32(a, zero_i32), max_a);
                b = _mm512_min_epi32(_mm512_max_epi32(b, zero_i32), max_b);
                const __m512i product = _mm512_mullo_epi32(a, b);
                const __m512i scaled = _mm512_srli_epi32(
                    _mm512_add_epi32(
                        _mm512_mullo_epi32(product, _mm512_loadu_si512(
                            reinterpret_cast<const void*>(multiplier.data() + j))),
                        rounding_q15),
                    15);
                const __m512i clipped = _mm512_min_epi32(
                    _mm512_max_epi32(scaled, zero_i32), max_u7);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + j),
                                 _mm512_cvtusepi32_epi8(clipped));
            }
        };
        make_pairs(us_src, w.fc_us_clip_max, w.fc_us_pair_q15, x_q8.data());
        make_pairs(them_src, w.fc_them_clip_max, w.fc_them_pair_q15,
                   x_q8.data() + PAIRWISE_HALF);
#elif defined(__AVX2__)
        const __m256i zero_i32 = _mm256_setzero_si256();
        const __m256i max_u7 = _mm256_set1_epi32(127);
        const __m256i rounding_q15 = _mm256_set1_epi32(1 << 14);
        const auto load_i16_as_i32 = [](const std::int16_t* ptr) {
            return _mm256_cvtepi16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
        };
        const auto pack_u8 = [](__m256i values) {
            const __m128i lo = _mm256_castsi256_si128(values);
            const __m128i hi = _mm256_extracti128_si256(values, 1);
            return _mm_packus_epi16(
                _mm_packus_epi32(lo, hi), _mm_setzero_si128());
        };
        const auto make_pairs = [&](const AccumulatorArray& source,
                                    const std::array<std::int16_t, EMB_DIM>& clip,
                                    const std::array<std::int32_t, PAIRWISE_HALF>& multiplier,
                                    std::uint8_t* destination) {
            for (std::size_t j = 0; j < PAIRWISE_HALF; j += 8) {
                __m256i a = load_i16_as_i32(source.data() + j);
                __m256i b = load_i16_as_i32(source.data() + j + PAIRWISE_HALF);
                const __m256i max_a = load_i16_as_i32(clip.data() + j);
                const __m256i max_b = load_i16_as_i32(clip.data() + j + PAIRWISE_HALF);
                a = _mm256_min_epi32(_mm256_max_epi32(a, zero_i32), max_a);
                b = _mm256_min_epi32(_mm256_max_epi32(b, zero_i32), max_b);
                const __m256i product = _mm256_mullo_epi32(a, b);
                const __m256i scaled = _mm256_srli_epi32(
                    _mm256_add_epi32(
                        _mm256_mullo_epi32(
                            product,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                multiplier.data() + j))),
                        rounding_q15),
                    15);
                const __m256i clipped = _mm256_min_epi32(
                    _mm256_max_epi32(scaled, zero_i32), max_u7);
                _mm_storel_epi64(
                    reinterpret_cast<__m128i*>(destination + j), pack_u8(clipped));
            }
        };
        make_pairs(us_src, w.fc_us_clip_max, w.fc_us_pair_q15, x_q8.data());
        make_pairs(them_src, w.fc_them_clip_max, w.fc_them_pair_q15,
                   x_q8.data() + PAIRWISE_HALF);
#else
        for (std::size_t j = 0; j < PAIRWISE_HALF; ++j) {
            const auto pair_to_u7 = [](std::int16_t a, std::int16_t b,
                                       std::int16_t max_a, std::int16_t max_b,
                                       std::int32_t multiplier) {
                const std::int32_t ca = std::clamp<std::int32_t>(a, 0, max_a);
                const std::int32_t cb = std::clamp<std::int32_t>(b, 0, max_b);
                const std::int32_t q = (ca * cb * multiplier + (1 << 14)) >> 15;
                return static_cast<std::uint8_t>(std::clamp(q, 0, 127));
            };
            x_q8[j] = pair_to_u7(
                us_src[j], us_src[j + PAIRWISE_HALF],
                w.fc_us_clip_max[j], w.fc_us_clip_max[j + PAIRWISE_HALF],
                w.fc_us_pair_q15[j]);
            x_q8[j + PAIRWISE_HALF] = pair_to_u7(
                them_src[j], them_src[j + PAIRWISE_HALF],
                w.fc_them_clip_max[j], w.fc_them_clip_max[j + PAIRWISE_HALF],
                w.fc_them_pair_q15[j]);
        }
#endif
        }
        }

        // First layer after the pairwise product: int8 matmul + bias + ReLU in one kernel.
        {
#if !ANDYBOT_AVX512 && defined(__AVX2__) && !(defined(__AVX512F__) || defined(_M_AVX512F))
        if (nnz_count >= 0) project_l1_sparse_avx2(hd.l1_weights.data(), x_q8.data(),
            nnz_index.data(), nnz_count, hd.l1_scale.data(), hd.l1_bias.data(), h2.data());
        else
#endif
#if ANDYBOT_AVX512 && (defined(__AVX512F__) || defined(_M_AVX512F))
        if (nnz_count >= 0) project_l1_sparse(hd.l1_weights.data(), x_q8.data(),
            nnz_index.data(), nnz_count, hd.l1_scale.data(), hd.l1_bias.data(), h2.data());
        else
#endif
        project_l1(hd.l1_weights.data(), x_q8.data(),
            hd.l1_scale.data(), hd.l1_bias.data(), h2.data());
        }

        {
#if defined(__AVX512F__) || defined(_M_AVX512F)
            // Same arithmetic as below with the sums split into independent chains (the scalar version is one
            // 128-long chain of dependent float adds). Only float rounding order differs.
            float value = 0.0f;
            {
                const __m512 z = _mm512_setzero_ps(), one = _mm512_set1_ps(1.0f);
                const __m512 h0 = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(h2.data()), z), one);
                const __m512 h1 = _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(h2.data() + 16), z), one);
                alignas(64) float first[64];
                _mm512_store_ps(first, h0); _mm512_store_ps(first + 16, h1);
                _mm512_store_ps(first + 32, _mm512_mul_ps(h0, h0)); _mm512_store_ps(first + 48, _mm512_mul_ps(h1, h1));
                __m512 a0 = _mm512_loadu_ps(hd.l2_bias.data()), b0 = _mm512_loadu_ps(hd.l2_bias.data() + 16);
                __m512 a1 = z, b1 = z, a2 = z, b2 = z, a3 = z, b3 = z;
                const float* W = hd.l2_weights_t.data();
                for (std::size_t i = 0; i < 64; i += 4) {
                    const __m512 x0 = _mm512_set1_ps(first[i]), x1 = _mm512_set1_ps(first[i + 1]);
                    const __m512 x2 = _mm512_set1_ps(first[i + 2]), x3 = _mm512_set1_ps(first[i + 3]);
                    a0 = _mm512_fmadd_ps(x0, _mm512_loadu_ps(W + i * 32), a0);
                    b0 = _mm512_fmadd_ps(x0, _mm512_loadu_ps(W + i * 32 + 16), b0);
                    a1 = _mm512_fmadd_ps(x1, _mm512_loadu_ps(W + i * 32 + 32), a1);
                    b1 = _mm512_fmadd_ps(x1, _mm512_loadu_ps(W + i * 32 + 48), b1);
                    a2 = _mm512_fmadd_ps(x2, _mm512_loadu_ps(W + i * 32 + 64), a2);
                    b2 = _mm512_fmadd_ps(x2, _mm512_loadu_ps(W + i * 32 + 80), b2);
                    a3 = _mm512_fmadd_ps(x3, _mm512_loadu_ps(W + i * 32 + 96), a3);
                    b3 = _mm512_fmadd_ps(x3, _mm512_loadu_ps(W + i * 32 + 112), b3);
                }
                const __m512 s0 = _mm512_min_ps(_mm512_max_ps(
                    _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)), z), one);
                const __m512 s1 = _mm512_min_ps(_mm512_max_ps(
                    _mm512_add_ps(_mm512_add_ps(b0, b1), _mm512_add_ps(b2, b3)), z), one);
                const float* O = hd.out_weights.data();
                __m512 t = _mm512_mul_ps(h0, _mm512_loadu_ps(O));
                t = _mm512_fmadd_ps(h1, _mm512_loadu_ps(O + 16), t);
                __m512 u = _mm512_mul_ps(_mm512_mul_ps(h0, h0), _mm512_loadu_ps(O + 32));
                u = _mm512_fmadd_ps(_mm512_mul_ps(h1, h1), _mm512_loadu_ps(O + 48), u);
                t = _mm512_fmadd_ps(s0, _mm512_loadu_ps(O + 64), t);
                u = _mm512_fmadd_ps(s1, _mm512_loadu_ps(O + 80), u);
                t = _mm512_fmadd_ps(_mm512_mul_ps(s0, s0), _mm512_loadu_ps(O + 96), t);
                u = _mm512_fmadd_ps(_mm512_mul_ps(s1, s1), _mm512_loadu_ps(O + 112), u);
                value = hd.out_bias + _mm512_reduce_add_ps(_mm512_add_ps(t, u));
            }
            if (false) {
#elif !ANDYBOT_AVX512 && defined(__AVX2__) && defined(__FMA__)
            float value = 0.0f;
            {
                const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
                __m256 h[4];
                alignas(32) float first[64];
                for (int k = 0; k < 4; ++k) {
                    h[k] = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(h2.data() + 8 * k), z), one);
                    _mm256_store_ps(first + 8 * k, h[k]);
                    _mm256_store_ps(first + 32 + 8 * k, _mm256_mul_ps(h[k], h[k]));
                }
                __m256 acc[4][4];  // [chain][eight outputs]
                for (int q = 0; q < 4; ++q) {
                    acc[0][q] = _mm256_loadu_ps(hd.l2_bias.data() + 8 * q);
                    acc[1][q] = z; acc[2][q] = z; acc[3][q] = z;
                }
                const float* W = hd.l2_weights_t.data();
                for (std::size_t i = 0; i < 64; i += 4)
                    for (int c = 0; c < 4; ++c) {
                        const __m256 x = _mm256_set1_ps(first[i + c]);
                        const float* row = W + (i + c) * 32;
                        acc[c][0] = _mm256_fmadd_ps(x, _mm256_loadu_ps(row), acc[c][0]);
                        acc[c][1] = _mm256_fmadd_ps(x, _mm256_loadu_ps(row + 8), acc[c][1]);
                        acc[c][2] = _mm256_fmadd_ps(x, _mm256_loadu_ps(row + 16), acc[c][2]);
                        acc[c][3] = _mm256_fmadd_ps(x, _mm256_loadu_ps(row + 24), acc[c][3]);
                    }
                __m256 sv[4];
                for (int q = 0; q < 4; ++q)
                    sv[q] = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(_mm256_add_ps(acc[0][q], acc[1][q]),
                                                                      _mm256_add_ps(acc[2][q], acc[3][q])), z), one);
                const float* O = hd.out_weights.data();
                __m256 tl = _mm256_mul_ps(h[0], _mm256_loadu_ps(O)), th = _mm256_mul_ps(h[1], _mm256_loadu_ps(O + 8));
                tl = _mm256_fmadd_ps(h[2], _mm256_loadu_ps(O + 16), tl);
                th = _mm256_fmadd_ps(h[3], _mm256_loadu_ps(O + 24), th);
                __m256 ul = _mm256_mul_ps(_mm256_mul_ps(h[0], h[0]), _mm256_loadu_ps(O + 32));
                __m256 uh = _mm256_mul_ps(_mm256_mul_ps(h[1], h[1]), _mm256_loadu_ps(O + 40));
                ul = _mm256_fmadd_ps(_mm256_mul_ps(h[2], h[2]), _mm256_loadu_ps(O + 48), ul);
                uh = _mm256_fmadd_ps(_mm256_mul_ps(h[3], h[3]), _mm256_loadu_ps(O + 56), uh);
                tl = _mm256_fmadd_ps(sv[0], _mm256_loadu_ps(O + 64), tl);
                th = _mm256_fmadd_ps(sv[1], _mm256_loadu_ps(O + 72), th);
                ul = _mm256_fmadd_ps(sv[2], _mm256_loadu_ps(O + 80), ul);
                uh = _mm256_fmadd_ps(sv[3], _mm256_loadu_ps(O + 88), uh);
                tl = _mm256_fmadd_ps(_mm256_mul_ps(sv[0], sv[0]), _mm256_loadu_ps(O + 96), tl);
                th = _mm256_fmadd_ps(_mm256_mul_ps(sv[1], sv[1]), _mm256_loadu_ps(O + 104), th);
                ul = _mm256_fmadd_ps(_mm256_mul_ps(sv[2], sv[2]), _mm256_loadu_ps(O + 112), ul);
                uh = _mm256_fmadd_ps(_mm256_mul_ps(sv[3], sv[3]), _mm256_loadu_ps(O + 120), uh);
                // Sum of the sixteen lanes: halves, quarters, pairs.
                const __m256 r8 = _mm256_add_ps(_mm256_add_ps(tl, ul), _mm256_add_ps(th, uh));
                __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
                r4 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
                r4 = _mm_add_ss(r4, _mm_shuffle_ps(r4, r4, 1));
                value = hd.out_bias + _mm_cvtss_f32(r4);
            }
            if (false) {
#else
            float value = 0.0f;
            {
#endif
            std::array<float, 64> first{};
            std::array<float, 32> second{};
            for (std::size_t i = 0; i < 32; ++i) {
                first[i] = std::clamp(h2[i], 0.0f, 1.0f);
                first[32 + i] = first[i] * first[i];
            }
            // Keep each output's addition order, but reuse each broadcast
            // across all outputs and expose independent dependency chains.
#if defined(__AVX512F__) || defined(_M_AVX512F)
            auto a = _mm512_loadu_ps(hd.l2_bias.data());
            auto b = _mm512_loadu_ps(hd.l2_bias.data() + 16);
            for (std::size_t i = 0; i < 64; ++i) {
                const auto x = _mm512_set1_ps(first[i]);
                const auto row = hd.l2_weights_t.data() + i * 32;
#if !ANDYBOT_AVX512
                a = _mm512_fmadd_ps(x, _mm512_loadu_ps(row), a);
                b = _mm512_fmadd_ps(x, _mm512_loadu_ps(row + 16), b);
#else
                a = _mm512_add_ps(a, _mm512_mul_ps(x, _mm512_loadu_ps(row)));
                b = _mm512_add_ps(b, _mm512_mul_ps(x, _mm512_loadu_ps(row + 16)));
#endif
            }
            _mm512_storeu_ps(second.data(), a);
            _mm512_storeu_ps(second.data() + 16, b);
#else
            auto a = _mm256_loadu_ps(hd.l2_bias.data());
            auto b = _mm256_loadu_ps(hd.l2_bias.data() + 8);
            auto c = _mm256_loadu_ps(hd.l2_bias.data() + 16);
            auto d = _mm256_loadu_ps(hd.l2_bias.data() + 24);
            for (std::size_t i = 0; i < 64; ++i) {
                const auto x = _mm256_set1_ps(first[i]);
                const auto row = hd.l2_weights_t.data() + i * 32;
#if !ANDYBOT_AVX512
                a = _mm256_fmadd_ps(x, _mm256_loadu_ps(row), a);
                b = _mm256_fmadd_ps(x, _mm256_loadu_ps(row + 8), b);
                c = _mm256_fmadd_ps(x, _mm256_loadu_ps(row + 16), c);
                d = _mm256_fmadd_ps(x, _mm256_loadu_ps(row + 24), d);
#else
                a = _mm256_add_ps(a, _mm256_mul_ps(x, _mm256_loadu_ps(row)));
                b = _mm256_add_ps(b, _mm256_mul_ps(x, _mm256_loadu_ps(row + 8)));
                c = _mm256_add_ps(c, _mm256_mul_ps(x, _mm256_loadu_ps(row + 16)));
                d = _mm256_add_ps(d, _mm256_mul_ps(x, _mm256_loadu_ps(row + 24)));
#endif
            }
            _mm256_storeu_ps(second.data(), a); _mm256_storeu_ps(second.data() + 8, b);
            _mm256_storeu_ps(second.data() + 16, c); _mm256_storeu_ps(second.data() + 24, d);
#endif
            value = hd.out_bias;
            for (std::size_t i = 0; i < 64; ++i)
                value += first[i] * hd.out_weights[i];
            for (std::size_t i = 0; i < 32; ++i) {
                const float c = std::clamp(second[i], 0.0f, 1.0f);
                value += c * hd.out_weights[64 + i];
                value += c * c * hd.out_weights[96 + i];
            }
            }
            return static_cast<int>(std::clamp(value * 410.0f, -10000.0f, 10000.0f));
        }
    }
