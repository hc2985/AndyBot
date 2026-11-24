// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.
//
// Portions of this file are derived from Berserk 13, Copyright (C) 2024 Jay Honnold, licensed under the GNU
// General Public License version 3 or later; ported to C++ and modified for AndyBot in November 2026.

    // =========================================================================================================
    // AndyBot's main search and quiescence search.
    //
    // The search began as a restructured port of Berserk 13's search (Jay Honnold, GPL-3). These rules have since
    // been replaced by AndyBot's own formulations with its own tuned numbers: futility (fail-soft floor), null move,
    // quiet-move pruning (one reach estimate), singular extensions (graded) and the history bonus. Still derived
    // from Berserk, constants included: reductions, late-move count, ProbCut, razoring, reverse futility, move
    // ordering stages, quiescence, table replacement, aspiration windows (root_window.hpp) and time management
    // (clock_plan.hpp). The data layout, the speed work and the network are AndyBot's; eval_norm puts the network's
    // score on a phase-stable scale.
    // Layout: table view and quiescence, then the main search as phases (gather, node cutoffs, move loop, store).
    // =========================================================================================================
#include "eval_norm.inc"

    static constexpr int S2_WIN = MATE - MAX_PLY;      // scores beyond this are mates
    static constexpr int S2_NO_EVAL = -INF;            // stack_ marker for "in check, no static eval"
    static constexpr std::uint8_t TT_NONE = 3;         // table entry that carries only a static eval / move

    // eval_norm for every |raw| below the last knot, per piece band (same arithmetic as the loop in eval_norm).
    static constexpr int kS2NormTableN = 2048;
    struct S2NormTable {
        std::int16_t v[5][kS2NormTableN]{};
        S2NormTable() {
            for (int band = 0; band < 5; ++band)
                for (int a = 0; a < kS2NormTableN; ++a) {
                    int x;
                    if (a >= kNormIn[band][kNormN - 1]) {
                        x = kNormOut[band][kNormN - 1] + (a - kNormIn[band][kNormN - 1]) * kNormTailX100[band] / 100;
                    } else {
                        int i = 1;
                        while (kNormIn[band][i] < a) ++i;
                        x = kNormOut[band][i - 1]
                          + (a - kNormIn[band][i - 1]) * (kNormOut[band][i] - kNormOut[band][i - 1])
                                / (kNormIn[band][i] - kNormIn[band][i - 1]);
                    }
                    v[band][a] = static_cast<std::int16_t>(x);
                }
        }
    };
    static inline const S2NormTable kS2NormTable{};
    // S2_NORM_PCT: one scale on the normalised eval (100 = the fitted curve as is).
    static int s2_norm_scale(int v) {
#if ANDYBOT_T_S2_NORM_PCT != 100
        return v * TP(S2_NORM_PCT) / 100;
#else
        return v;
#endif
    }
    int eval_norm(int raw) const {
        {
            const int fa = raw < 0 ? -raw : raw;
            if (fa < kS2NormTableN) {
                const int fp = bb::popcount(board_.occ);
                const int fb = fp >= 27 ? 0 : fp >= 21 ? 1 : fp >= 15 ? 2 : fp >= 9 ? 3 : 4;
                const int fv = s2_norm_scale(kS2NormTable.v[fb][fa]);
                return raw < 0 ? -fv : fv;
            }
        }
        const int pieces = bb::popcount(board_.occ);
        const int band = pieces >= 27 ? 0 : pieces >= 21 ? 1 : pieces >= 15 ? 2 : pieces >= 9 ? 3 : 4;
        const int a = raw < 0 ? -raw : raw;
        int v;
        if (a >= kNormIn[band][kNormN - 1]) {
            v = kNormOut[band][kNormN - 1] + (a - kNormIn[band][kNormN - 1]) * kNormTailX100[band] / 100;
        } else {
            int i = 1;
            while (kNormIn[band][i] < a) ++i;
            v = kNormOut[band][i - 1]
              + (a - kNormIn[band][i - 1]) * (kNormOut[band][i] - kNormOut[band][i - 1])
                    / (kNormIn[band][i] - kNormIn[band][i - 1]);
        }
        v = s2_norm_scale(v);
        return raw < 0 ? -v : v;
    }

    struct S2Tables {
        int lmr[64][64]{};
        int lmp[2][64]{};
        S2Tables() {
            for (int d = 1; d < 64; ++d)
                for (int m = 1; m < 64; ++m)
                    lmr[d][m] = static_cast<int>(std::log(double(d)) * std::log(double(m)) / (TP(S2_LMR_DIV_X10000) / 10000.0) + TP(S2_LMR_BASE_X10000) / 10000.0);
            lmr[1][0] = lmr[0][1] = 0;
            for (int d = 0; d < 64; ++d) {
                lmp[0][d] = static_cast<int>(1.3050 + (TP(S2_LMP_NONIMP_X10000) / 10000.0) * d * d);
                lmp[1][d] = static_cast<int>(2.1885 + (TP(S2_LMP_IMP_X10000) / 10000.0) * d * d);
            }
        }
    };
    static const S2Tables& s2_tables() {
        static const S2Tables t;
        return t;
    }
    static int s2_history_bonus(int depth) { return std::min(TP(S2_HB_MAX), TP(S2_HB_LIN) * depth); }

    // A node's accumulator is built only when needed: its own eval, or just before a move is made from it.
    void s2_mat(int ply) { materialize_accumulator(ply); }
    bool s2_in_check() const { return incheck(board_, board_.stm); }
    // Reply table indexed by the piece that just moved and its target square (0 = none).
    int s2_counters_[12][64] = {};
    int s2_counter_at(int ply) const {
        if (ply <= 0) return -1;
        const Stack& p = stack_[ply - 1];
        if (p.movedPiece < 0 || p.currentMove.to < 0) return -1;
        return s2_counters_[p.movedPiece][p.currentMove.to];
    }
    // Running average score per root move (index = move key), INT_MIN = not searched yet this game move.
    static constexpr int S2_ROOT_KEYS = (4 << 12) | (63 << 6) | 63;
    std::vector<int> s2_root_avg_ = std::vector<int>(S2_ROOT_KEYS + 1, INT_MIN);
    // Time management inputs: nodes spent under each root move in the current search, and the score of the previous search.
    std::vector<std::uint64_t> s2_root_nodes_ = std::vector<std::uint64_t>(S2_ROOT_KEYS + 1, 0);
    int s2_prev_search_score_ = INT_MIN;
    int s2_root_first_key_ = -1;  // the root's first move (best move of the previous root search), -1 = use the table
    int s2_nmp_min_ply_ = 0;
    int s2_null_count_ = 0;       // null moves on the current line (OPT_REP_BOUND)
    int s2_null_ply_ = -100000;   // ply of the node right after the latest null move on the current line

    // Upcoming repetition. Every reversible move of a non-pawn piece changes the position key by a fixed amount
    // (both squares of the piece plus the side to move). A table of those amounts, two-slot cuckoo hashed, tells
    // whether some earlier position on the line is one such move away from the current one.
    struct ReversibleMoves {
        static constexpr std::size_t kSlots = 8192;
        std::array<std::uint64_t, kSlots> diff{};
        std::array<std::uint8_t, kSlots> sq_a{}, sq_b{};
        static std::size_t slot1(std::uint64_t d) { return static_cast<std::size_t>(d & (kSlots - 1)); }
        static std::size_t slot2(std::uint64_t d) { return static_cast<std::size_t>((d >> 16) & (kSlots - 1)); }
        ReversibleMoves() {
            const char pieces[] = "NBRQKnbrqk";
            for (int c = 0; c < 10; ++c) {
                const int kind = c % 5;
                const int zi = zobrist_piece_index(pieces[c]);
                for (int a = 0; a < 64; ++a) {
                    const bb::Bitboard reach = kind == 0 ? bb::KNIGHT_ATTACKS[a]
                        : kind == 1 ? bb::bishop_attacks(a, 0)
                        : kind == 2 ? bb::rook_attacks(a, 0)
                        : kind == 3 ? (bb::bishop_attacks(a, 0) | bb::rook_attacks(a, 0))
                        : bb::KING_ATTACKS[a];
                    for (int b = a + 1; b < 64; ++b)
                        if ((reach >> b) & 1) insert(ZOBRIST_PIECE[zi][a] ^ ZOBRIST_PIECE[zi][b] ^ ZOBRIST_STM, a, b);
                }
            }
        }
        // Cuckoo insertion: an occupant that is displaced moves to its other slot.
        void insert(std::uint64_t d, int a, int b) {
            std::uint8_t x = static_cast<std::uint8_t>(a), y = static_cast<std::uint8_t>(b);
            std::size_t i = slot1(d);
            while (true) {
                std::swap(diff[i], d);
                std::swap(sq_a[i], x);
                std::swap(sq_b[i], y);
                if (d == 0) return;
                i = (i == slot1(d)) ? slot2(d) : slot1(d);
            }
        }
        // The squares of the reversible move that changes the key by `d`, if there is one.
        bool find(std::uint64_t d, int& a, int& b) const {
            std::size_t i = slot1(d);
            if (diff[i] != d) {
                i = slot2(d);
                if (diff[i] != d) return false;
            }
            a = sq_a[i];
            b = sq_b[i];
            return true;
        }
    };
    static const ReversibleMoves& reversible_moves() {
        static const ReversibleMoves table;
        return table;
    }
    // Key of the position `back` plies before the current one (`ply` plies into the search), from the search line
    // and then the game record. False when the record does not reach that far.
    bool key_plies_ago(int back, int ply, std::uint64_t& out) const {
        if (back <= ply) { out = search_path_hashes_[ply - back]; return true; }
        const int idx = static_cast<int>(game_history_hashes_.size()) - 1 - (back - ply);
        if (idx < 0) return false;
        out = game_history_hashes_[static_cast<std::size_t>(idx)];
        return true;
    }
    // Can the side to move reach an earlier position with one reversible move? Positions inside the search line
    // count at once; positions from before the root must already have occurred twice.
    bool s2_upcoming_repetition(int ply) const {
        const int span = std::min(static_cast<int>(board_.halfmove), ply - s2_null_ply_);
        if (span < 3 || ply >= MAX_PLY) return false;
        const ReversibleMoves& table = reversible_moves();
        for (int back = 3; back <= span; back += 2) {
            std::uint64_t then;
            if (!key_plies_ago(back, ply, then)) break;
            int a, b;
            if (!table.find(board_.hash ^ then, a, b)) continue;
            if (LINE_TABLES.between[a][b] & board_.occ) continue;  // the path must be open
            if (ply > back) return true;
            const int piece = piece_index(board_.sq[a]) >= 0 ? piece_index(board_.sq[a]) : piece_index(board_.sq[b]);
            if (piece < 0 || (piece < 6 ? WHITE : (WHITE ^ 1)) != board_.stm) continue;
            for (int earlier = back + 4; earlier <= span; earlier += 2) {
                std::uint64_t k;
                if (!key_plies_ago(earlier, ply, k)) break;
                if (k == then) return true;
            }
        }
        return false;
    }
    // Insufficient material: no pawns, rooks or queens, at most four pieces in all (kings included), and each side
    // has nothing, one knight, two knights or one bishop. No such ending can be won by force.
    bool s2_material_draw() const {
        if (bb::popcount(board_.occ) > 4) return false;
        for (int side = 0; side < 2; ++side) {
            const auto& p = board_.pieces[side];
            if (p[0] | p[3] | p[4]) return false;
            const int knights = bb::popcount(p[1]), bishops = bb::popcount(p[2]);
            if (bishops > 1 || (bishops == 1 && knights > 0)) return false;
        }
        return true;
    }
    int s2_root_best_key_ = -1;   // best root move of the current root search, set when a move beats alpha
    int s2_root_best_score_ = 0;
    // Correction history: running average of (search score - static eval) per pawn structure, in 1/256 eval units.
    std::vector<std::int16_t> s2_corr_ = std::vector<std::int16_t>(131072, 0);
    std::size_t s2_corr_index() const {
        return static_cast<std::size_t>(((board_.pieces[WHITE][0] * 0x9E3779B97F4A7C15ULL)
                                         ^ (board_.pieces[BLACK][0] * 0xC2B2AE3D27D4EB4FULL)) >> 47);
    }
    int s2_nmp_color_ = WHITE;

    // Table store and probe for this search. An entry for the same position keeps its move when the new result has
    // none, and is replaced only by an exact score or a result not much shallower than the stored one. Otherwise the
    // entry with the lowest depth, less a penalty for every search since it was written, gives way. A hit refreshes
    // the entry's search number.
    static constexpr int S2_AGE_SHIFT = 61;
    static constexpr int kAgeDepthPenalty = 4;   // per search since the entry was written
    static constexpr int kKeepDepthMargin = 4;   // a same-position result this much shallower keeps the stored one
    std::uint8_t s2_tt_age_ = 0;                 // search number, low 3 bits stored in each entry
    std::uint64_t s2_age_bits() const { return static_cast<std::uint64_t>(s2_tt_age_ & 7) << S2_AGE_SHIFT; }
    int s2_rel_age(std::uint64_t d) const {
        return static_cast<int>((8 + (s2_tt_age_ & 7) - ((d >> S2_AGE_SHIFT) & 7)) & 7);
    }
    std::uint64_t pack_entry(int depth, int score, int key, int raw_eval, std::uint8_t bound, bool pv, int ply) const {
        const int stored = andybot::search_score::to_tt(score, ply, MATE, MAX_PLY);
        return tt_pack(depth, stored, key, raw_eval, bound) | (static_cast<std::uint64_t>(pv) << 60) | s2_age_bits();
    }
    void tt_store2(std::uint64_t h, int depth, int score, std::uint8_t bound, int best_key, int raw_eval, bool pv,
                   int ply) {
        if (!tt_) return;
        TTBucket& bucket = tt_[static_cast<std::size_t>(h) & tt_bucket_mask_];
        int slot = 0, weakest = INT_MAX;
        std::uint64_t same = 0;
        bool found = false;
        for (int i = 0; i < TT_BUCKET_N; ++i) {
            const std::uint64_t d = tt_word_load(bucket.data[i]);
            if (d == 0) {  // the first empty slot wins over any occupied one
                if (weakest != INT_MIN) { slot = i; weakest = INT_MIN; }
                continue;
            }
            if ((tt_word_load(bucket.key_xor_data[i]) ^ d) == h) { slot = i; found = true; same = d; break; }
            const int worth = tt_packed_depth(d) - kAgeDepthPenalty * s2_rel_age(d);
            if (worth < weakest) { weakest = worth; slot = i; }
        }
        int key = best_key >= 0 ? best_key : 0xFFFF;
        std::uint64_t packed;
        if (found) {
            TTEntry old;
            tt_unpack(same, old);
            if (key == 0xFFFF) key = old.best_key;
            if (bound == TT_EXACT || depth + kKeepDepthMargin > old.depth) {
                packed = pack_entry(depth, score, key, raw_eval, bound, pv, ply);
            } else {
                if (key == old.best_key) return;  // nothing new to record
                packed = tt_pack(old.depth, old.score, key, old.static_eval, old.bound) | (same & (1ULL << 60)) | s2_age_bits();
            }
        } else {
            packed = pack_entry(depth, score, key, raw_eval, bound, pv, ply);
        }
        if (packed == 0) packed = 1ULL << 26;  // never write the "empty" pattern (move key 1 is no real move)
        tt_word_store(bucket.data[slot], packed);
        tt_word_store(bucket.key_xor_data[slot], h ^ packed);
    }
    bool tt_probe2(std::uint64_t h, TTEntry& out, bool& pv, int ply) {
        if (!tt_) return false;
        TTBucket& bucket = tt_[static_cast<std::size_t>(h) & tt_bucket_mask_];
        for (int i = 0; i < TT_BUCKET_N; ++i) {
            const std::uint64_t d = tt_word_load(bucket.data[i]);
            if (d == 0 || (tt_word_load(bucket.key_xor_data[i]) ^ d) != h) continue;
            tt_unpack(d, out);
            out.score = andybot::search_score::from_tt(out.score, ply, MATE, MAX_PLY);
            pv = ((d >> 60) & 1) != 0;
            if (s2_rel_age(d) != 0) {  // touched in this search: refresh its search number
                const std::uint64_t fresh = (d & ~(7ULL << S2_AGE_SHIFT)) | s2_age_bits();
                tt_word_store(bucket.data[i], fresh);
                tt_word_store(bucket.key_xor_data[i], h ^ fresh);
            }
            return true;
        }
        return false;
    }

    bool s2_legal(const Move& m, bool in_chk, bb::Bitboard pinned, bb::Bitboard checkers) {
        if (m.ep) {
            Undo u;
            ::makemove(board_, m, u);
            const bool ok = !incheck(board_, board_.stm ^ 1);
            ::unmakemove(board_, u);
            return ok;
        }
        if (m.castle) return true;  // the generator and the table-move decoder already verified the king's path
        return in_chk ? legal_in_check(board_, m, pinned, checkers) : legal_when_not_in_check(board_, m, pinned);
    }

    // Static eval of the node in normalised units with the 50-move fade, and the raw value for the table.
    int s2_pre_fade_eval_ = 0;
    int s2_static_eval(int ply, bool tt_hit, const TTEntry& tte, int& raw) {
        raw = (tt_hit && tte.static_eval != TT_NO_STATIC_EVAL) ? tte.static_eval : eval_cp_unscaled(ply);
        int v = eval_norm(raw);
        v = std::clamp(v + s2_corr_[s2_corr_index()] / 256 / 2, -S2_WIN + 1, S2_WIN - 1);
        s2_pre_fade_eval_ = v;
        return v * (200 - std::min(board_.halfmove, 100)) / 200;
    }

    // Exchange test for pruning and ordering: after the best sequence of recaptures on the target square, does the
    // move keep at least `threshold`? Castling, en passant and promotions always pass (not modelled).
    static constexpr int kS2SeeValue[7] = {100, 422, 422, 642, 1015, 30000, 0};
    static int s2_piece_value(char piece) {
        const int code = piece_index(piece);
        return code < 0 ? 0 : kS2SeeValue[code % 6];
    }
    // Cheapest piece of `side` attacking `to` through `occ` (rays given), as a kind 0..5, or -1.
    int cheapest_attacker(int side, int to, bb::Bitboard occ, bb::Bitboard diagonal, bb::Bitboard straight,
                          bb::Bitboard& from_mask) const {
        const bb::Bitboard (&pc)[6] = board_.pieces[side];
        auto take = [&](bb::Bitboard found, int kind) { from_mask = bb::sq_bb(bb::lsb(found)); return kind; };
        bb::Bitboard f;
        if ((f = bb::PAWN_ATTACKS[side ^ 1][to] & pc[0] & occ)) return take(f, 0);
        if ((f = bb::KNIGHT_ATTACKS[to] & pc[1] & occ)) return take(f, 1);
        if ((f = diagonal & pc[2] & occ)) return take(f, 2);
        if ((f = straight & pc[3] & occ)) return take(f, 3);
        if ((f = (diagonal | straight) & pc[4] & occ)) return take(f, 4);
        if ((f = bb::KING_ATTACKS[to] & pc[5] & occ)) return take(f, 5);
        return -1;
    }
    bool s2_see_ge(const Move& m, int threshold) const {
        if (m.castle || m.ep || m.promo) return true;
        const Board& b = board_;
        const int mover = piece_index(b.sq[m.from]);
        if (mover < 0) return 0 >= threshold;
        const int to = m.to;
        const int victim = piece_index(b.sq[to]);
        // `balance`: how far the side that just captured is above what it needs, seen from the side to recapture.
        int balance = (victim < 0 ? 0 : kS2SeeValue[victim % 6]) - threshold;
        if (balance < 0) return false;           // even an uncontested capture falls short
        balance = kS2SeeValue[mover % 6] - balance;
        if (balance <= 0) return true;           // even losing the mover keeps the threshold
        bb::Bitboard occ = (b.occ & ~bb::sq_bb(m.from)) | bb::sq_bb(to);
        const bb::Bitboard diag_rays = bb::bishop_attacks(to, 0), line_rays = bb::rook_attacks(to, 0);
        bb::Bitboard diagonal = bb::bishop_attacks(to, occ), straight = bb::rook_attacks(to, occ);
        int side = b.stm ^ 1;
        bool we_hold = true;                     // with no recapture left, the initial capture stands
        for (int guard = 0; guard < 30; ++guard) {
            bb::Bitboard from_mask = 0;
            const int kind = cheapest_attacker(side, to, occ, diagonal, straight, from_mask);
            if (kind < 0) break;
            we_hold = !we_hold;
            balance = kS2SeeValue[kind] - balance;
            if (balance < static_cast<int>(we_hold)) break;
            occ &= ~from_mask;
            if (from_mask & diag_rays) diagonal = bb::bishop_attacks(to, occ);   // x-rays behind the capturer
            if (from_mask & line_rays) straight = bb::rook_attacks(to, occ);
            side ^= 1;
        }
        return we_hold;
    }

    // Continuation history as one table [earlier move was a capture][its piece, to][this piece, to],
    // shared by every lag (the per-lag tables of the first search are not used).
    std::vector<std::int16_t> s2_ch_ = std::vector<std::int16_t>(static_cast<std::size_t>(2) * CMH_DIM * CMH_DIM, 0);
    const std::int16_t* s2_ch_row(int prev_ply) const {
        if (prev_ply < 0) return nullptr;
        const Stack& p = stack_[prev_ply];
        if (p.movedPiece < 0 || p.currentMove.to < 0) return nullptr;
        const std::size_t row = ((p.currentMove.cap || p.currentMove.ep) ? CMH_DIM : 0)
                              + static_cast<std::size_t>(p.movedPiece * 64 + p.currentMove.to);
        return s2_ch_.data() + row * CMH_DIM;
    }
    int s2_ch(int prev_ply, int col) const {
        const std::int16_t* r = s2_ch_row(prev_ply);
        return r ? r[col] : 0;
    }
    // The four rows a node reads (1, 2, 4, 6 plies back), looked up once per node; an all-zero row stands in for a
    // missing one.
    static inline const std::int16_t s2_zero_row_[768]{};
    struct S2Rows { const std::int16_t* r[4]; };
    S2Rows s2_rows(int ply) const {
        S2Rows x;
        for (std::size_t i = 0; i < 4; ++i) {
            const std::int16_t* q = s2_ch_row(ply - CONTINUATION_LAGS[i]);
            x.r[i] = q ? q : s2_zero_row_;
        }
        return x;
    }
    // ---------------------------------------------------------------------------------------------------------------
    // Move ordering. A stage table per mode; each stage generates and scores its moves when first reached.
    //   main:        table move, good noisy, killer 1, killer 2, counter, quiets (+ under-promotions), bad noisy
    //   quiescence:  noisy moves by score (under-promotions are never tried here)
    //   evasion:     table move, noisy by victim, quiets (+ under-promotions)
    // A noisy move is "good" when the exchange keeps at least minus half its ordering score.
    // ---------------------------------------------------------------------------------------------------------------
    template <std::size_t N> struct RawMoves {  // storage that is not initialised; only written slots are read
        union U { Move d[N]; U() noexcept {} ~U() noexcept {} } u;
        Move& operator[](std::size_t i) noexcept { return u.d[i]; }
        const Move& operator[](std::size_t i) const noexcept { return u.d[i]; }
    };
    using CaptureBuf = RawMoves<32>;
    using UnderBuf = RawMoves<24>;
    class MoveOrder {
    public:
        MoveOrder(Engine& eng, int ply, int tt_move, bool in_qsearch, bool in_check, const Move* tt_decoded = nullptr)
            : e_(eng), ply_(ply), tt_(tt_move) {
            if (tt_decoded) { tt_known_ = true; tt_ok_ = tt_decoded->from >= 0; if (tt_ok_) tt_move_ = *tt_decoded; }
            if (!in_qsearch) {
                stages_ = kMain;
                k1_ = ply < MAX_PLY ? e_.killers_[ply][0] : -1;
                k2_ = ply < MAX_PLY ? e_.killers_[ply][1] : -1;
                counter_ = e_.s2_counter_at(ply);
            } else {
                stages_ = in_check ? kEvasion : kQuiescence;
            }
        }
        void skip_quiets() { no_quiets_ = true; }
        bool next(Move& out) {
            while (true) {
                const Stage st = stages_[at_];
                if (st == Stage::Done) return false;
                if (!ready_) { prepare(st); ready_ = true; }
                if (produce(st, out)) {
                    if (single_shot(st)) { ++at_; ready_ = false; }
                    return true;
                }
                ++at_;
                ready_ = false;
            }
        }

    private:
        enum class Stage : std::uint8_t { TableMove, GoodNoisy, Killer1, Killer2, Counter, Quiets, BadNoisy,
                                          QsNoisy, EvasionNoisy, EvasionQuiets, Done };
        static constexpr Stage kMain[] = {Stage::TableMove, Stage::GoodNoisy, Stage::Killer1, Stage::Killer2,
                                          Stage::Counter, Stage::Quiets, Stage::BadNoisy, Stage::Done};
        static constexpr Stage kQuiescence[] = {Stage::QsNoisy, Stage::Done};
        static constexpr Stage kEvasion[] = {Stage::TableMove, Stage::EvasionNoisy, Stage::EvasionQuiets, Stage::Done};

        void prepare(Stage st) {
            switch (st) {
            case Stage::GoodNoisy:
            case Stage::QsNoisy:
                load_noisy(0);
                for (std::size_t i = 0; i < end_; ++i) score_[i] = noisy_score(moves_[i]);
                deferred_ = 0;
                break;
            case Stage::EvasionNoisy:
                load_noisy(0);
                for (std::size_t i = 0; i < end_; ++i) {
                    const Move& c = moves_[i];
                    score_[i] = 10000000 + (c.ep ? kS2SeeValue[0] : s2_piece_value(e_.board_.sq[c.to]));
                }
                break;
            case Stage::Quiets:        // after the deferred noisy moves, which stay at the front
                load_quiets(deferred_);
                break;
            case Stage::EvasionQuiets:
                load_quiets(0);
                break;
            case Stage::BadNoisy:
                cur_ = 0;
                end_ = deferred_;
                break;
            default:
                break;
            }
        }
        bool produce(Stage st, Move& out) {
            switch (st) {
            case Stage::TableMove:
                if (tt_known_) { if (tt_ok_) out = tt_move_; return tt_ok_; }
                return tt_ >= 0 && decode_tt_move(e_.board_, tt_, false, out);
            case Stage::GoodNoisy:
                while (cur_ != end_) {
                    pick_best();
                    const Move c = moves_[cur_];
                    const int sc = score_[cur_];
                    ++cur_;
                    if (move_key(c) == tt_) continue;
                    if (e_.s2_see_ge(c, -sc / 2)) { out = c; return true; }
                    moves_[deferred_] = c;  // slots before cur_ are free: keep it for the last stage
                    score_[deferred_] = sc;
                    ++deferred_;
                }
                return false;
            case Stage::Killer1:
                return !no_quiets_ && k1_ >= 0 && k1_ != tt_ && quiet_from_key(k1_, out);
            case Stage::Killer2:
                return !no_quiets_ && k2_ >= 0 && k2_ != tt_ && k2_ != k1_ && quiet_from_key(k2_, out);
            case Stage::Counter:
                return !no_quiets_ && counter_ > 0 && counter_ != tt_ && counter_ != k1_ && counter_ != k2_
                       && quiet_from_key(counter_, out);
            case Stage::Quiets:
                while (cur_ != end_ && !no_quiets_) {
                    pick_best();
                    const Move q = moves_[cur_++];
                    const int key = move_key(q);
                    if (key != tt_ && key != k1_ && key != k2_ && key != counter_) { out = q; return true; }
                }
                return false;
            case Stage::BadNoisy:
                while (cur_ != end_) {
                    const Move c = moves_[cur_++];
                    if (move_key(c) != tt_) { out = c; return true; }
                }
                return false;
            case Stage::QsNoisy:
                if (cur_ == end_) return false;
                pick_best();
                out = moves_[cur_++];
                return true;
            case Stage::EvasionNoisy:
                while (cur_ != end_) {
                    pick_best();
                    const Move c = moves_[cur_++];
                    if (move_key(c) != tt_) { out = c; return true; }
                }
                return false;
            case Stage::EvasionQuiets:
                while (cur_ != end_ && !no_quiets_) {
                    pick_best();
                    const Move q = moves_[cur_++];
                    if (move_key(q) != tt_) { out = q; return true; }
                }
                return false;
            default:
                return false;
            }
        }
        static bool single_shot(Stage st) {
            return st == Stage::TableMove || st == Stage::Killer1 || st == Stage::Killer2 || st == Stage::Counter;
        }

        // Captures and promotions; under-promotions are held back for the quiet stage.
        void load_noisy(std::size_t keep) {
            moves_.resize(keep);
            genpseudo_tactical(e_.board_, moves_);
            std::size_t w = keep;
            const std::size_t n = moves_.size();
            for (std::size_t i = keep; i < n; ++i) {
                const Move c = moves_[i];
                if (c.promo && (c.promo | 32) != 'q') { if (n_under_ < 24) under_[n_under_++] = c; }
                else moves_[w++] = c;
            }
            moves_.resize(w);
            cur_ = keep;
            end_ = w;
        }
        // Quiet moves (unless quiets are off) plus the held-back under-promotions, scored, after `keep` moves.
        void load_quiets(std::size_t keep) {
            cur_ = end_ = keep;
            if (no_quiets_) return;
            moves_.resize(keep);
            genpseudo_quiet(e_.board_, moves_);
            for (std::size_t i = 0; i < n_under_; ++i) moves_.push_back(under_[i]);
            end_ = moves_.size();
            for (std::size_t i = cur_; i < end_; ++i) score_[i] = quiet_score(moves_[i]);
        }
        // Highest remaining score to the front; the earliest one wins ties.
        void pick_best() {
            std::size_t best = cur_;
#if defined(__AVX512F__)
            const std::size_t n = end_ - cur_;
            constexpr std::size_t kSimdPickMin = 8;  // below this the scalar scan is faster
            if (n >= kSimdPickMin) {
                const int* sc = score_ + cur_;
                __m512i vmax = _mm512_set1_epi32(INT_MIN);
                std::size_t i = 0;
                for (; i + 16 <= n; i += 16) vmax = _mm512_max_epi32(vmax, _mm512_loadu_si512(sc + i));
                if (i < n) {
                    const __mmask16 k = static_cast<__mmask16>((1u << (n - i)) - 1);
                    vmax = _mm512_mask_max_epi32(vmax, k, vmax, _mm512_maskz_loadu_epi32(k, sc + i));
                }
                const __m512i top = _mm512_set1_epi32(_mm512_reduce_max_epi32(vmax));
                for (i = 0; i < n; i += 16) {
                    const __mmask16 k = i + 16 <= n ? static_cast<__mmask16>(0xFFFF)
                                                    : static_cast<__mmask16>((1u << (n - i)) - 1);
                    const __mmask16 eq = _mm512_mask_cmpeq_epi32_mask(k, _mm512_maskz_loadu_epi32(k, sc + i), top);
                    if (eq) { best = cur_ + i + static_cast<std::size_t>(__builtin_ctz(eq)); break; }
                }
            } else
#endif
            for (std::size_t i = cur_ + 1; i < end_; ++i)
                if (score_[i] > score_[best]) best = i;
            std::swap(moves_[cur_], moves_[best]);
            std::swap(score_[cur_], score_[best]);
        }
        // Victim value plus a sixteenth of the capture history (en passant counts as no victim).
        int noisy_score(const Move& c) const {
            const int victim = c.ep ? 0 : s2_piece_value(e_.board_.sq[c.to]);
            return ((c.cap || c.ep) ? e_.capture_history_score_at(c, ply_) / 16 : 0) + victim;
        }
        // Main history and continuations (weights 2, 2, 1, 1 for 1, 2, 4, 6 plies back), plus a bonus for moving a
        // piece out of reach of a cheaper attacker and a penalty for moving into it.
        int quiet_score(const Move& q) {
            const int piece = piece_index(e_.board_.sq[q.from]);
            int sc = 2 * e_.quiet_hist(e_.board_.stm, ply_, q.from, q.to);
            if (piece >= 0) {
                const int col = piece * 64 + q.to;
                if (!rows_ready_) { rows_ = e_.s2_rows(ply_); rows_ready_ = true; }
                sc += 2 * rows_.r[0][col] + 2 * rows_.r[1][col] + rows_.r[2][col] + rows_.r[3][col];
            }
            const int kind = piece % 6;
            if (kind >= 1 && kind <= 4) {
                const Stack& st = e_.stack_[ply_];
                const std::uint64_t reach = kind <= 2 ? st.threatPawn : kind == 3 ? st.threatMinor : st.threatRook;
                if ((reach >> q.from) & 1) sc += 16384;
                if ((reach >> q.to) & 1) sc -= 16384;
            }
            return sc;
        }
        bool quiet_from_key(int key, Move& out) const {
            return decode_tt_move(e_.board_, key, false, out) && !(out.cap || out.ep || out.promo);
        }

        Engine& e_;
        int ply_, tt_;
        bool tt_known_ = false, tt_ok_ = false;  // table move decoded by the caller
        Move tt_move_{};
        const Stage* stages_ = kMain;
        std::size_t at_ = 0;
        bool ready_ = false;
        int k1_ = -1, k2_ = -1, counter_ = -1;
        bool no_quiets_ = false;
        bool rows_ready_ = false;
        S2Rows rows_{};
        UnderBuf under_;
        std::size_t n_under_ = 0;
        std::size_t cur_ = 0, end_ = 0, deferred_ = 0;
        MoveList moves_{};
        int score_[MoveList::CAPACITY];
    };
    void s2_cont_update(int ply, int piece, int to, int bonus) {
        if (piece < 0 || to < 0) return;
        for (const int lag : CONTINUATION_LAGS) {
            const std::int16_t* r = s2_ch_row(ply - lag);
            if (r) gravity_update(const_cast<std::int16_t&>(r[piece * 64 + to]), bonus);
        }
    }
    // History of a move for pruning decisions: capture history, or the quiet table plus continuations 1, 2, 4 back.
    int s2_move_history(const Move& m, int ply, int side, int piece) const {
        if (m.cap || m.ep) return capture_history_score(m);
        int v = quiet_hist(side, ply, m.from, m.to);
        if (piece >= 0) {
            const int col = piece * 64 + m.to;
            v += s2_ch(ply - 1, col) + s2_ch(ply - 2, col) + s2_ch(ply - 4, col);
        }
        return v;
    }

    // History update on a cutoff: credit the cutting move, blame every move of the same kind tried before it.
    void credit_capture(const Move& c, int amount) {
        const int attacker = piece_index(board_.sq[c.from]);
        const int victim = piece_index(c.ep ? (board_.stm == WHITE ? 'p' : 'P') : board_.sq[c.to]);
        if (attacker >= 0 && victim >= 0) gravity_update(cap_hist(attacker, c.to, victim), amount);
    }
    void credit_quiet(const Move& q, int ply, int amount) {
        gravity_update(quiet_hist(board_.stm, ply, q.from, q.to), amount);
        s2_cont_update(ply, piece_index(board_.sq[q.from]), q.to, amount);
    }
    // Killers for this ply, and the reply table keyed by the opponent's last move.
    void remember_refutation(int key, int ply) {
        if (ply >= 0 && ply < MAX_PLY && killers_[ply][0] != key) {
            killers_[ply][1] = killers_[ply][0];
            killers_[ply][0] = key;
        }
        if (ply > 0 && stack_[ply - 1].movedPiece >= 0 && stack_[ply - 1].currentMove.to >= 0)
            s2_counters_[stack_[ply - 1].movedPiece][stack_[ply - 1].currentMove.to] = key;
    }
    void s2_update_histories(const Move& best, int ply, int depth, const MoveList& quiets,
                             const CaptureBuf& captures, int n_captures) {
        const int amount = s2_history_bonus(depth);
        const int best_key = move_key(best);
        if (best.cap || best.ep) {
            credit_capture(best, amount);
        } else {
            const bool queen_promo = best.promo && promo_code(best.promo) == promo_code('q');
            if (!queen_promo) remember_refutation(best_key, ply);
            // A cutoff by the only quiet move tried, at low depth, teaches little.
            if (quiets.size() > 1 || depth > 4) credit_quiet(best, ply, amount);
            for (const Move& q : quiets)
                if (move_key(q) != best_key) credit_quiet(q, ply, -amount);
        }
        for (int i = 0; i < n_captures; ++i)
            if (move_key(captures[i]) != best_key) credit_capture(captures[i], -amount);
    }

    // Quiet moves that give a direct check (no discovered checks): queens, rooks, bishops, knights, then pawns.
    static int s2_pop_ref(bb::Bitboard& swapped) { return bb::pop_lsb(swapped) ^ 56; }  // squares from rank 8 down
    void s2_quiet_checks(MoveList& out) const {
        const Board& b = board_;
        const int us = b.stm, them = us ^ 1;
        const int ek = b.king[them];
        const bb::Bitboard occ = b.occ;
        const bb::Bitboard bmask = bb::bishop_attacks(ek, occ), rmask = bb::rook_attacks(ek, occ);
        const bb::Bitboard masks[5] = {0, bb::KNIGHT_ATTACKS[ek], bmask, rmask, bmask | rmask};
        for (int kind = 4; kind >= 1; --kind) {
            bb::Bitboard movers = __builtin_bswap64(b.pieces[us][kind]);
            while (movers) {
                const int from = s2_pop_ref(movers);
                const bb::Bitboard att = kind == 1 ? bb::KNIGHT_ATTACKS[from]
                                       : kind == 2 ? bb::bishop_attacks(from, occ)
                                       : kind == 3 ? bb::rook_attacks(from, occ)
                                       : (bb::bishop_attacks(from, occ) | bb::rook_attacks(from, occ));
                bb::Bitboard targets = __builtin_bswap64(att & masks[kind] & ~occ);
                while (targets) out.push_back(Move{from, s2_pop_ref(targets), 0, false, false, false, false});
            }
        }
        const bb::Bitboard opts = bb::PAWN_ATTACKS[them][ek];
        const bb::Bitboard pawns = b.pieces[us][0];
        const bb::Bitboard promo_from = us == WHITE ? 0x00FF000000000000ULL : 0x000000000000FF00ULL;
        const bb::Bitboard dp_rank = us == WHITE ? 0x0000000000FF0000ULL : 0x0000FF0000000000ULL;
        const auto fwd = [us](bb::Bitboard x) { return us == WHITE ? x << 8 : x >> 8; };
        const int dir = us == WHITE ? 8 : -8;
        const bb::Bitboard single = fwd(pawns & ~promo_from) & ~occ;
        bb::Bitboard t1 = __builtin_bswap64(single & opts);
        bb::Bitboard t2 = __builtin_bswap64(fwd(single & dp_rank) & ~occ & opts);
        while (t1) { const int to = s2_pop_ref(t1); out.push_back(Move{to - dir, to, 0, false, false, false, false}); }
        while (t2) { const int to = s2_pop_ref(t2); out.push_back(Move{to - 2 * dir, to, 0, false, false, false, true}); }
        bb::Bitboard t3 = __builtin_bswap64(fwd(pawns & promo_from) & ~occ & opts);
        while (t3) {
            const int to = s2_pop_ref(t3);
            for (const char pr : {'r', 'b', 'n'})
                out.push_back(Move{to - dir, to, us == WHITE ? static_cast<char>(pr - 32) : pr, false, false, false, false});
        }
    }
    // ---------------------------------------------------------------------------------------------------------------
    // Table view: what the transposition table knows about the current position.
    // ---------------------------------------------------------------------------------------------------------------
    struct TableView {
        TTEntry e{};
        bool hit = false;        // an entry exists
        bool pv = false;         // the entry or the current node is on a principal line
        bool has_score = false;  // the entry carries a searched score, not only a static eval
        int move = -1;           // stored move key, -1 = none
    };
    TableView look_up(std::uint64_t h, int ply, bool pv_node) {
        TableView t;
        t.hit = tt_probe2(h, t.e, t.pv, ply);
        t.pv = t.pv || pv_node;
        t.move = t.hit && t.e.best_key != 0xFFFF ? t.e.best_key : -1;
        t.has_score = t.hit && t.e.bound != TT_NONE;
        return t;
    }
    // True when the stored bound is valid on the side where the stored score lies (`above`: the score is above the
    // reference value, so a lower bound is needed; otherwise an upper bound). Exact scores hold on both sides.
    static bool bound_holds(const TTEntry& e, bool above) {
        return e.bound == TT_EXACT || e.bound == (above ? TT_LOWER : TT_UPPER);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Quiescence: resolve captures (and every evasion when in check) until the position is quiet; at the first two
    // quiescence plies, also direct quiet checks.
    // ---------------------------------------------------------------------------------------------------------------
    // Fifty-move rule (a mate still counts) or material that cannot mate.
    bool rule_draw(bool checked) {
        if (board_.halfmove >= 100 && (!checked || !genlegal(board_, false).empty())) return true;
        return s2_material_draw();
    }

    // Requests the hash bucket of the position after an ordinary move while the move is still being set up.
    void prefetch_child(const Move& m, int moved_piece) const {
        if (moved_piece < 0 || m.promo || m.castle || m.ep) return;
        std::uint64_t k = board_.hash ^ ZOBRIST_STM ^ ZOBRIST_PIECE[moved_piece][m.from] ^ ZOBRIST_PIECE[moved_piece][m.to];
        hash_toggle_ep(k, board_.ep);
        const int victim = piece_index(board_.sq[m.to]);
        if (victim >= 0) k ^= ZOBRIST_PIECE[victim][m.to];
        tt_prefetch(k);
    }

    // Plays `m`, searches the child in quiescence and takes it back. A quiet move that repeats a position scores 0.
    int quiesce_child(const Move& m, int moved_piece, bool capture, int alpha, int beta, int ply, int qply) {
        Undo u;
        stack_[ply].currentMove = m;
        stack_[ply].movedPiece = moved_piece;
        prefetch_child(m, moved_piece);
        s2_mat(ply);
        make_move(m, u);
        const std::uint64_t child = board_.hash;
        tt_prefetch(child);
        int score = 0;
        if (capture || !is_repetition_hash(child, ply)) {
            search_path_hashes_[std::min(ply + 1, MAX_PLY + 1)] = child;
            search_path_moves_[std::min(ply + 1, MAX_PLY + 1)] = m;
            prepare_child_accumulator(ply, m, u);
            score = -quiesce(-beta, -alpha, ply + 1, qply - 1);
        }
        unmake_move(u);
        return score;
    }

    int quiesce(int alpha, int beta, int ply, int qply = 0) {
        nodes_++;
        if (ply > sel_depth_) sel_depth_ = ply;
        if (stop_requested_now()) return 0;
        const bool pv_node = beta - alpha != 1;
        const bool checked = s2_in_check();
        if (ply >= MAX_PLY) return checked ? 0 : eval_norm(eval_cp_unscaled(ply));
        if (rule_draw(checked)) return 0;
        const std::uint64_t h = board_.hash;
        search_path_hashes_[std::min(ply, MAX_PLY + 1)] = h;

        const TableView tv = look_up(h, ply, pv_node);
        if (!pv_node && tv.has_score && bound_holds(tv.e, tv.e.score >= beta)) return tv.e.score;

        // Stand pat: the side to move may decline every capture.
        int raw = TT_NO_STATIC_EVAL;
        int best = -MATE + ply;
        int futility = -MATE;
        if (checked) {
            stack_[ply].staticEval = S2_NO_EVAL;
        } else {
            int stand = s2_static_eval(ply, tv.hit, tv.e, raw);
            stack_[ply].staticEval = s2_pre_fade_eval_;
            if (tv.has_score && bound_holds(tv.e, tv.e.score > stand)) stand = tv.e.score;
            if (!tv.hit) tt_store2(h, -1, 0, TT_NONE, -1, raw, tv.pv, ply);
            if (stand >= beta) return stand;
            alpha = std::max(alpha, stand);
            best = stand;
            futility = stand + TP(S2_QS_FUT);
        }
        stack_[ply].inCheck = checked;
        if (checked) compute_threats(ply);

        MoveOrder picker(*this, ply, tv.move, /*in_qsearch=*/true, checked);
        const bb::Bitboard pinned = pinned_pieces(board_);
        const bb::Bitboard checkers = checked ? checkers_of(board_) : 0;
        MoveList tried_quiets;                 // for the history update on a cutoff
        CaptureBuf tried_captures{};
        int n_tried_captures = 0;
        int legal = 0, best_key = -1;
        bool stopped = false;

        // Searches one move and updates the bounds; true when the node is finished (cutoff or stop).
        auto consider = [&](const Move& m, int moved_piece, bool capture) {
            const int score = quiesce_child(m, moved_piece, capture, alpha, beta, ply, qply);
            if (stop_requested_.load(std::memory_order_relaxed)) { stopped = true; return true; }
            if (checked && score > -S2_WIN) picker.skip_quiets();  // one evasion that holds: no more quiet ones
            if (score <= best) return false;
            best = score;
            if (score > alpha) {
                best_key = move_key(m);
                alpha = score;
            }
            if (alpha < beta) return false;
            s2_update_histories(m, ply, 1, tried_quiets, tried_captures, n_tried_captures);
            return true;
        };

        // Phase 1: captures and promotions (all evasions when in check).
        bool done = false;
        Move m;
        while (picker.next(m)) {
            if (!s2_legal(m, checked, pinned, checkers)) continue;
            ++legal;
            const bool capture = m.cap || m.ep;
            prefetch_child(m, piece_index(board_.sq[m.from]));  // as soon as the move is legal, before the pruning tests
            if (best > -S2_WIN) {
                // When even winning the exchange outright cannot reach alpha, a move must gain material; otherwise
                // it must not lose any.
                const bool futile_node = !checked && futility <= alpha;
                if (!s2_see_ge(m, futile_node ? 1 : 0)) {
                    if (futile_node) best = std::max(best, futility);
                    continue;
                }
            }
            const int moved_piece = piece_index(board_.sq[m.from]);
            if (!capture && !m.promo) tried_quiets.push_back(m);
            else if (capture && n_tried_captures < 32) tried_captures[n_tried_captures++] = m;
            if (consider(m, moved_piece, capture)) { done = true; break; }
        }
        if (stopped) return 0;

        // Phase 2: quiet direct checks, first two quiescence plies only.
        if (!done && !checked && qply >= -1 && alpha < beta) {
            MoveList checks;
            s2_quiet_checks(checks);
            for (const Move& m : checks) {
                if (!s2_legal(m, false, pinned, 0)) continue;
                if (best > -S2_WIN && !s2_see_ge(m, 0)) continue;
                tried_quiets.push_back(m);
                if (consider(m, piece_index(board_.sq[m.from]), false)) break;
            }
            if (stopped) return 0;
        }

        if (legal == 0 && checked) return -MATE + ply;
        if (best >= beta && std::abs(best) < S2_WIN) best = (best + beta) / 2;  // soften fail-high scores
        tt_store2(h, 0, best, best >= beta ? TT_LOWER : TT_UPPER, best_key, raw, tv.pv, ply);
        return best;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Main search
    // ---------------------------------------------------------------------------------------------------------------
    // Everything the phases of one node share.
    struct Node {
        int ply = 0, depth = 0, alpha = 0, beta = 0, alpha_orig = 0;
        bool pv = false, cut = false, checked = false;
        bool excluded = false;       // a singular-extension probe: one move is left out
        int excluded_key = -1;
        TableView tv;
        int tt_move = -1;            // move to try first: the table's, or at the root the best move so far
        int eval = 0, raw = TT_NO_STATIC_EVAL;
        bool improving = false;
        int stm = WHITE;
        int easy = -1;               // opponent can win material with a cheaper piece: -1 unknown, 0/1
        int legal_seen = 0;          // legal moves met so far in the move loop
    };
    bool opponent_easy_capture(Node& n) {
        if (n.easy < 0) n.easy = lesser_piece_threat(n.stm) ? 1 : 0;
        return n.easy != 0;
    }
    static int draw_score(std::uint64_t node_count) { return 2 - static_cast<int>(node_count & 3); }

    // Static eval, refined by the table, and whether it is better than two (or four) plies ago.
    void node_eval(Node& n) {
        Stack& ss = stack_[n.ply];
        ss.inCheck = n.checked;
        if (n.checked) {
            ss.staticEval = S2_NO_EVAL;
            return;
        }
        n.eval = s2_static_eval(n.ply, n.tv.hit, n.tv.e, n.raw);
        ss.staticEval = s2_pre_fade_eval_;
        // A searched score is a better estimate than the static eval when its bound points that way.
        if (n.tv.has_score && bound_holds(n.tv.e, n.tv.e.score > n.eval)) n.eval = n.tv.e.score;
        if (!n.tv.hit && !n.excluded) tt_store2(board_.hash, -1, 0, TT_NONE, -1, n.raw, n.tv.pv, n.ply);
        if (n.ply >= 2) {
            const int two = stack_[n.ply - 2].staticEval;
            const int ref = (n.ply >= 4 && two == S2_NO_EVAL) ? stack_[n.ply - 4].staticEval : two;
            n.improving = ref == S2_NO_EVAL || ss.staticEval > ref;
        }
    }

    // --- node-level cutoffs (non-PV, not in check) -------------------------------------------------------------------
    // Eval far above beta: assume the node holds, unless the table move is a quiet with poor history.
    bool reverse_futility(Node& n, int& out) {
        if (n.depth > 9 || n.excluded || n.eval >= S2_WIN || n.eval < n.beta) return false;
        const int margin = TP(S2_RFP_DEPTH) * n.depth - TP(S2_RFP_IMPROVING) * (n.improving && !opponent_easy_capture(n));
        if (n.eval - margin < n.beta) return false;
        if (n.tt_move >= 0) {
            Move tm;
            if (!decode_tt_move(board_, n.tt_move, false, tm)) return false;
            if (s2_move_history(tm, n.ply, n.stm, piece_index(board_.sq[tm.from])) <= 11800) return false;
        }
        out = (n.eval + n.beta) / 2;
        return true;
    }
    // Eval far below alpha: let quiescence confirm that nothing rescues the node.
    bool razor(Node& n, int& out) {
        if (n.depth > 5 || n.eval + TP(S2_RAZOR) * n.depth > n.alpha) return false;
        const int s = quiesce(n.alpha, n.beta, n.ply);
        if (stop_requested_.load(std::memory_order_relaxed)) { out = 0; return true; }
        if (s > n.alpha) return false;
        out = s;
        return true;
    }
    // Pass the move; if the opponent still cannot reach beta, the node is almost certainly a cut. Deep cuts are
    // verified by a reduced search with null moves suspended for this side.
    bool null_move(Node& n, int& out) {
        const int ply = n.ply, depth = n.depth;
        const bool after_null = ply > 0 && stack_[ply - 1].currentMove.to < 0;
        // Passing must look clearly good enough: the static eval has to clear beta by a margin.
        if (depth < 4 || after_null || n.excluded || n.eval < n.beta + TP(S2_NMP_MARGIN) || opponent_easy_capture(n)) return false;
        if (!has_non_pawn_material(n.stm) || (ply < s2_nmp_min_ply_ && n.stm == s2_nmp_color_)) return false;
        const int R = (TP(S2_NMP_R0) + TP(S2_NMP_DEPTH_X1024) * depth
                       + std::min(TP(S2_NMP_EVAL_X1024) * (n.eval - n.beta), TP(S2_NMP_EVAL_CAP))) / 1024;  // summed in 1/1024 ply
        Stack& ss = stack_[ply];
        const int saved_ep = board_.ep;
        const std::uint64_t saved_hash = board_.hash;
        hash_toggle_ep(board_.hash, board_.ep);
        board_.ep = -1;
        board_.stm ^= 1;
        hash_toggle_stm(board_.hash);
        s2_mat(ply);
        if (model_loaded_) prepare_null_accumulator(ply);
        search_path_hashes_[std::min(ply + 1, MAX_PLY + 1)] = board_.hash;
        search_path_moves_[std::min(ply + 1, MAX_PLY + 1)] = Move{};
        const Move saved_move = ss.currentMove;
        const int saved_piece = ss.movedPiece;
        ss.currentMove = Move{};
        ss.movedPiece = -1;
        tt_prefetch(board_.hash);
        const int saved_null_ply = s2_null_ply_;
        s2_null_ply_ = ply + 1;
        ++s2_null_count_;
        int s = -search_node(depth - R, -n.beta, -n.beta + 1, ply + 1, !n.cut);
        --s2_null_count_;
        s2_null_ply_ = saved_null_ply;
        ss.currentMove = saved_move;
        ss.movedPiece = saved_piece;
        board_.stm ^= 1;
        board_.ep = saved_ep;
        board_.hash = saved_hash;
        if (stop_requested_.load(std::memory_order_relaxed)) { out = 0; return true; }
        if (s < n.beta) return false;
        if (s >= S2_WIN) s = n.beta;  // never trust a mate found by passing
        if (s2_nmp_min_ply_ != 0 || (std::abs(n.beta) < S2_WIN && depth < 14)) { out = s; return true; }
        s2_nmp_min_ply_ = ply + 3 * (depth - R) / 4;
        s2_nmp_color_ = n.stm;
        const int verify = search_node(depth - R, n.beta - 1, n.beta, ply, false);
        s2_nmp_min_ply_ = 0;
        if (stop_requested_.load(std::memory_order_relaxed)) { out = 0; return true; }
        if (verify < n.beta) return false;
        out = s;
        return true;
    }
    // A capture that holds well above beta in quiescence and then in a reduced search makes the node a cut.
    bool probcut(Node& n, int& out) {
        const int ply = n.ply, depth = n.depth;
        const int pc_beta = n.beta + TP(S2_PROBCUT);
        if (depth < 6 || n.excluded || std::abs(n.beta) >= S2_WIN) return false;
        if (n.tv.has_score && n.tv.e.depth >= depth - 3 && n.tv.e.score < pc_beta) return false;
        const int see_needed = pc_beta > n.eval ? 1 : 0;
        // Captures and queen promotions, best victim first (promotions ahead).
        MoveList cand;
        genpseudo_tactical(board_, cand);
        std::size_t kept = 0;
        for (std::size_t i = 0; i < cand.size(); ++i)
            if (!cand[i].promo || (cand[i].promo | 32) == 'q') cand[kept++] = cand[i];
        cand.resize(kept);
        int value[MoveList::CAPACITY];
        for (std::size_t i = 0; i < kept; ++i) {
            const Move& c = cand[i];
            value[i] = (c.ep ? kS2SeeValue[0] : s2_piece_value(board_.sq[c.to])) + (c.promo ? 2000 : 0);
        }
        const bb::Bitboard pinned = pinned_pieces(board_);
        Stack& ss = stack_[ply];
        for (std::size_t i = 0; i < kept; ++i) {
            std::size_t top = i;
            for (std::size_t j = i + 1; j < kept; ++j) if (value[j] > value[top]) top = j;
            std::swap(cand[i], cand[top]);
            std::swap(value[i], value[top]);
            if (!s2_see_ge(cand[i], see_needed)) continue;
            const Move c = cand[i];
            if (!s2_legal(c, false, pinned, 0)) continue;
            Undo u;
            ss.currentMove = c;
            ss.movedPiece = piece_index(board_.sq[c.from]);
            s2_mat(ply);
            make_move(c, u);
            tt_prefetch(board_.hash);
            search_path_hashes_[std::min(ply + 1, MAX_PLY + 1)] = board_.hash;
            search_path_moves_[std::min(ply + 1, MAX_PLY + 1)] = c;
            prepare_child_accumulator(ply, c, u);
            int s = -quiesce(-pc_beta, -pc_beta + 1, ply + 1);
            if (s >= pc_beta) s = -search_node(depth - 4, -pc_beta, -pc_beta + 1, ply + 1, !n.cut);
            unmake_move(u);
            if (stop_requested_.load(std::memory_order_relaxed)) { out = 0; return true; }
            if (s >= pc_beta) { out = s; return true; }
        }
        return false;
    }

    // --- move loop helpers ---------------------------------------------------------------------------------------------
    // History used for pruning and reductions: capture history, or the quiet table plus continuations 1, 2, 4 back.
    int pruning_history(const Move& m, bool capture, int moved_piece, const Node& n, const S2Rows& rows) const {
        if (capture) return capture_history_score_at(m, n.ply);
        int v = quiet_hist(n.stm, n.ply, m.from, m.to);
        if (moved_piece >= 0) {
            const int col = moved_piece * 64 + m.to;
            v += rows.r[0][col] + rows.r[1][col] + rows.r[2][col];
        }
        return v;
    }

    // Singular-extension probe for the table move. Returns the extension; sets `cut` (with `out`) when the probe
    // proves the whole node fails high, and `stopped` when the search was stopped.
    int singular_extension(Node& n, const Move& m, bool capture, bool& cut, int& out, bool& stopped) {
        const TTEntry& e = n.tv.e;
        const int s_beta = std::max(e.score - TP(S2_SE_BETA_X64) * n.depth / 64, -MATE);
        Stack& ss = stack_[n.ply];
        ss.excludedMove = m;
        const int s = search_node((n.depth - 1) * TP(S2_SE_PROBE_X64) / 64, s_beta - 1, s_beta, n.ply, n.cut);
        ss.excludedMove = Move{};
        if (stop_requested_.load(std::memory_order_relaxed)) { stopped = true; return 0; }
        const int parent_dext = n.ply > 0 ? stack_[n.ply - 1].dext : 0;
        if (s < s_beta) {  // nothing else reaches s_beta: one ply, plus more the further the alternatives fall short
            int ext = 1;
            if (!n.pv && ss.dext <= 6) {
                ext = std::min(1 + (s_beta - s) * TP(S2_SE_GAIN) / 1024, capture ? 2 : 3);
                if (ext > 1) ss.dext = parent_dext + 1;
            }
            return ext;
        }
        if (s_beta >= n.beta) { cut = true; out = s_beta; return 0; }  // several moves beat beta
        // Not singular: the table move is searched shallower, more so the further the alternatives got past the bar.
        if (e.score >= n.beta || n.cut) return n.pv ? -1 : -std::min(2, 1 + (s - s_beta) / TP(S2_SE_NEG_DIV));
        return e.score <= n.alpha ? -1 : 0;
    }

    // Searches the child after `m` has been made: reduced null window, re-search, principal-variation search.
    int search_child(Node& n, const Move& m, bool capture, int moved_piece, bool killer_or_counter, int reduction,
                     int new_depth, int played, int best) {
        const int ply = n.ply, depth = n.depth;
        int s = 0;
        if (depth > 1 && n.legal_seen > 1 && !(n.pv && capture)) {
            int R = reduction;
            if (!n.tv.pv) R += 2;
            if (!n.improving) ++R;
            if (killer_or_counter) R -= 2;
            if (s2_in_check()) --R;  // the move gives check
            if (n.cut) R += 1 + !capture;
            if (n.tv.hit && n.tv.e.depth >= depth) --R;
            R = std::min(new_depth, std::max(R, 1));
            const int reduced = new_depth - R;
            s = -search_node(reduced, -n.alpha - 1, -n.alpha, ply + 1, true);
            if (s > n.alpha && R > 1) {
                new_depth += s > best + TP(S2_LMR_DEEPER);
                new_depth -= s < best + new_depth;
                if (new_depth - 1 > reduced) s = -search_node(new_depth - 1, -n.alpha - 1, -n.alpha, ply + 1, !n.cut);
                if (!capture && !stop_requested_.load(std::memory_order_relaxed)) {
                    const int bonus = s <= n.alpha ? -s2_history_bonus(new_depth - 1)
                                    : s >= n.beta ? s2_history_bonus(new_depth - 1) : 0;
                    if (bonus != 0) s2_cont_update(ply, moved_piece, m.to, bonus);
                }
            }
        } else if (!n.pv || played > 1) {
            s = -search_node(new_depth - 1, -n.alpha - 1, -n.alpha, ply + 1, !n.cut);
        }
        if (n.pv && (played == 1 || (s > n.alpha && (s < n.beta || ply == 0))))
            s = -search_node(new_depth - 1, -n.beta, -n.alpha, ply + 1, false);
        return s;
    }

    // Table store and correction-history update at the end of a node.
    void store_node(const Node& n, int best, int best_key, bool best_is_capture) {
        if (n.excluded) return;
        const std::uint8_t bound = best >= n.beta ? TT_LOWER : best <= n.alpha_orig ? TT_UPPER : TT_EXACT;
        tt_store2(board_.hash, n.depth, best, bound, best_key, n.raw, n.tv.pv, n.ply);
        if (n.checked || n.raw == TT_NO_STATIC_EVAL || std::abs(best) >= S2_WIN) return;
        if (best_key >= 0 && best_is_capture) return;  // only quiet best moves
        const int static_value = eval_norm(n.raw);
        if (bound != TT_EXACT && bound != (best >= static_value ? TT_LOWER : TT_UPPER)) return;
        const int target = std::clamp((best - static_value) * 256, -30000, 30000);
        std::int16_t& slot = s2_corr_[s2_corr_index()];
        slot = static_cast<std::int16_t>((static_cast<int>(slot) * 255 + target) / 256);
    }

    int search_node(int depth, int alpha, int beta, int ply, bool cut_node) {
        // A repetition the side to move can force is worth at least a draw.
        if (ply > 0 && board_.halfmove >= 3 && alpha < 0 && s2_upcoming_repetition(ply)) {
            alpha = draw_score(static_cast<std::uint64_t>(nodes_));
            if (alpha >= beta) return alpha;
        }
        if (depth <= 0) return quiesce(alpha, beta, ply);
        if (stop_requested_now()) return 0;

        Node n;
        n.ply = ply;
        n.pv = beta - alpha != 1;
        n.cut = cut_node;
        n.checked = s2_in_check();
        if (ply >= MAX_PLY) return n.checked ? 0 : eval_norm(eval_cp_unscaled(ply));
        if (ply > 0 && board_.halfmove >= 100 && (!n.checked || !genlegal(board_, false).empty())) return 0;
        nodes_++;
        if (ply > 0 && s2_material_draw()) return draw_score(static_cast<std::uint64_t>(nodes_));
        // Mate distance: nothing here beats mating next move or loses to being mated now.
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;

        const std::uint64_t h = board_.hash;
        search_path_hashes_[std::min(ply, MAX_PLY + 1)] = h;
        Stack& ss = stack_[ply];
        n.excluded = ss.excludedMove.from >= 0;
        n.excluded_key = n.excluded ? move_key(ss.excludedMove) : -1;
        n.alpha = n.alpha_orig = alpha;
        n.beta = beta;
        n.depth = depth;
        n.stm = board_.stm;

        if (!n.excluded) n.tv = look_up(h, ply, n.pv);
        else n.tv.pv = n.pv;
        n.tt_move = (ply == 0 && !n.excluded && s2_root_first_key_ >= 0) ? s2_root_first_key_ : n.tv.move;
        if (!n.pv && n.tv.has_score && n.tv.e.depth >= depth && (cut_node || n.tv.e.score <= alpha)
            && bound_holds(n.tv.e, n.tv.e.score >= beta))
            return n.tv.e.score;

        node_eval(n);
        if (ply + 1 < MAX_PLY) { killers_[ply + 1][0] = -1; killers_[ply + 1][1] = -1; }
        stack_[ply + 1].excludedMove = Move{};
        ss.dext = ply > 0 ? stack_[ply - 1].dext : 0;
        // No move to start with at a node we expect to matter: search it one ply shallower.
        if (!n.excluded && !n.checked && (n.pv || cut_node) && n.depth >= 4 && n.tt_move < 0) --n.depth;

        if (!n.pv && !n.checked) {
            int out = 0;
            if (reverse_futility(n, out) || razor(n, out) || null_move(n, out) || probcut(n, out)) return out;
        }

        compute_threats(ply);
        const S2Rows rows = s2_rows(ply);
        Move tt_decoded{};  // from = -1 when there is no usable table move
        if (n.tt_move >= 0 && !decode_tt_move(board_, n.tt_move, false, tt_decoded)) tt_decoded = Move{};
        MoveOrder picker(*this, ply, n.tt_move, /*in_qsearch=*/false, n.checked, &tt_decoded);
        const bb::Bitboard pinned = pinned_pieces(board_);
        const bb::Bitboard checkers = n.checked ? checkers_of(board_) : 0;
        const int counter_key = ply > 0 ? s2_counter_at(ply) : -1;
        const bool tt_move_noisy = tt_decoded.from >= 0 && (tt_decoded.cap || tt_decoded.ep || tt_decoded.promo);
        const S2Tables& T = s2_tables();
        MoveList tried_quiets;
        CaptureBuf tried_captures{};
        int n_tried_captures = 0;
        int best = -MATE, best_key = -1, played = 0;
        bool best_is_capture = false;
        bool skip_quiets = false;
        int fut_floor = -INF;
        n.legal_seen = 0;
        Move m;
        while (picker.next(m)) {
            const int key = move_key(m);
            if (key == n.excluded_key) continue;
            if (!s2_legal(m, n.checked, pinned, checkers)) continue;
            ++n.legal_seen;
            const bool capture = m.cap || m.ep;
            if (skip_quiets && !capture && !m.promo) continue;
            const int moved_piece = piece_index(board_.sq[m.from]);
            prefetch_child(m, moved_piece);
            const bool killer_or_counter = key == killers_[ply][0] || key == killers_[ply][1] || key == counter_key;
            const int history = pruning_history(m, capture, moved_piece, n, rows);
            int reduction = T.lmr[std::min(n.depth, 63)][std::min(n.legal_seen, 63)];
            reduction -= history / TP(S2_LMR_HIST);
            reduction += tt_move_noisy;

            // Move pruning, once a score above a lost one is known.
            if (best > -S2_WIN) {
                if (ply > 0 && n.legal_seen >= T.lmp[n.improving][std::min(n.depth, 63)]) {
                    skip_quiets = true;
                    picker.skip_quiets();
                }
                const bool queen_promo = m.promo == 'Q' || m.promo == 'q';
                if (!capture && !queen_promo) {
                    // Quiets are judged at the depth they would actually be searched at.
                    const int lmr_depth = std::max(1, n.depth - reduction);
                    // One test: how high can this quiet plausibly lift the node? The static eval, an allowance per
                    // ply of remaining depth, and the record of the move. Not above alpha: later quiets are skipped,
                    // and this one too when its record is poor.
                    if (!n.checked && lmr_depth < TP(S2_PR_DEPTH)) {
                        const int allowance = TP(S2_FUT_BASE) + TP(S2_FUT_SLOPE) * lmr_depth;
                        const int reach = n.eval + allowance + (killer_or_counter ? 0 : history / TP(S2_PR_HIST_DIV));
                        if (reach <= n.alpha) {
                            fut_floor = std::max(fut_floor, std::min(reach, n.eval + allowance));
                            skip_quiets = true;
                            picker.skip_quiets();
                            if (history < 0 && !killer_or_counter) continue;
                        }
                    }
                    if (!s2_see_ge(m, -TP(S2_PR_SEEQ) * lmr_depth)) continue;
                } else if (!s2_see_ge(m, -TP(S2_SEEC_X100) * n.depth / 100 - std::clamp(history / TP(S2_PR_CAPH_DIV), -200, 200))) {
                    continue;
                }
            }
            ++played;
            if (!capture) { if (!m.promo) tried_quiets.push_back(m); }
            else if (n_tried_captures < 32) tried_captures[n_tried_captures++] = m;

            int extension = 0;
            if (ply > 0 && ply < 2 * root_depth_ && n.depth >= 6 && key == n.tt_move && n.tv.has_score
                && n.tv.e.depth >= n.depth - 3 && (n.tv.e.bound == TT_LOWER || n.tv.e.bound == TT_EXACT)
                && std::abs(n.tv.e.score) < S2_WIN) {
                bool cut = false, stopped = false;
                int out = 0;
                extension = singular_extension(n, m, capture, cut, out, stopped);
                if (stopped) return 0;
                if (cut) return out;
            }

            const std::uint64_t nodes_before = static_cast<std::uint64_t>(nodes_);
            Undo u;
            ss.currentMove = m;
            ss.movedPiece = moved_piece;
            prefetch_child(m, moved_piece);
            s2_mat(ply);
            make_move(m, u);
            const std::uint64_t child = board_.hash;
            tt_prefetch(child);
            search_path_hashes_[std::min(ply + 1, MAX_PLY + 1)] = child;
            int s;
            if (is_repetition_hash(child, ply)) {
                s = draw_score(static_cast<std::uint64_t>(nodes_));
            } else {
                search_path_moves_[std::min(ply + 1, MAX_PLY + 1)] = m;
                prepare_child_accumulator(ply, m, u);
                s = search_child(n, m, capture, moved_piece, killer_or_counter, reduction, n.depth + extension, played, best);
            }
            unmake_move(u);
            if (stop_requested_.load(std::memory_order_relaxed)) return 0;
            if (ply == 0 && key >= 0 && key <= S2_ROOT_KEYS) {
                s2_root_nodes_[static_cast<std::size_t>(key)] += static_cast<std::uint64_t>(nodes_) - nodes_before;
                int& avg = s2_root_avg_[static_cast<std::size_t>(key)];
                avg = avg == INT_MIN ? s : (avg + s) / 2;
            }
            if (s <= best) continue;
            best = s;
            if (s > n.alpha) {
                best_key = key;
                best_is_capture = capture;
                n.alpha = s;
                if (ply == 0) { s2_root_best_key_ = key; s2_root_best_score_ = s; }
                // The node now has a move that holds: look at the rest one ply less deep.
                if (n.alpha < n.beta && s > -S2_WIN) n.depth -= (n.depth >= 2 && n.depth <= 11);
            }
            if (n.alpha >= n.beta) {
                s2_update_histories(m, ply, n.depth + (best > n.beta + 77), tried_quiets, tried_captures, n_tried_captures);
                break;
            }
        }
        if (n.legal_seen == 0) return n.excluded ? n.alpha : n.checked ? -MATE + ply : 0;
        if (best == -MATE && played == 0) best = n.alpha_orig;  // everything was pruned: nothing proven beyond alpha
        if (best <= n.alpha_orig && best < fut_floor) best = std::min(fut_floor, n.alpha_orig);
        store_node(n, best, best_key, best_is_capture);
        return best;
    }
