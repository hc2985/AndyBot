// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.
//
// Portions of this file are derived from Berserk 13, Copyright (C) 2024 Jay Honnold, licensed under the GNU
// General Public License version 3 or later (threat map, 50-move fade); ported to C++ and modified for AndyBot in
// November 2026.

    // The static eval fades as the 50-move counter climbs (x (200 - halfmove) / 200, so half
    // at the limit). A side that is ahead then prefers lines that make progress (a pawn move or capture resets
    // the counter), and a side that is behind is drawn towards running it out. The eval cache and the table hold the
    // unscaled value, because neither key includes the counter.
    int fifty_scale(int value) const {
        return value * (200 - std::min(board_.halfmove, 100)) / 200;
    }
    int eval_cp(int ply) { return fifty_scale(eval_cp_unscaled(ply)); }
    int eval_cp_unscaled(int ply) {
        const std::size_t eval_cache_set = static_cast<std::size_t>(board_.hash)
            & ((std::size_t{1} << ANDYBOT_EVAL_CACHE_BITS) - 1);
        const std::size_t eval_cache_base = eval_cache_set * 2;
        for (std::size_t way = 0; way < 2; ++way) {
            const std::size_t index = eval_cache_base + way;
            const std::uint64_t bit = 1ULL << (index & 63);
            if ((eval_cache_valid_[index >> 6] & bit) != 0
                && eval_cache_keys_[index] == board_.hash)
                return static_cast<int>(eval_cache_values_[index]);
        }
        std::uint64_t& replace_word = eval_cache_replace_[eval_cache_set >> 6];
        const std::uint64_t replace_bit = 1ULL << (eval_cache_set & 63);
        const std::size_t replace_way = (replace_word & replace_bit) != 0 ? 1 : 0;
        replace_word ^= replace_bit;
        const std::size_t eval_cache_index = eval_cache_base + replace_way;
        const std::uint64_t eval_cache_bit = 1ULL << (eval_cache_index & 63);
        std::uint64_t& eval_cache_valid_word = eval_cache_valid_[eval_cache_index >> 6];
        const auto cache_eval = [&](int value) noexcept {
            const int clamped = std::clamp(value, -32000, 32000);
            eval_cache_keys_[eval_cache_index] = board_.hash;
            eval_cache_values_[eval_cache_index] = static_cast<std::int16_t>(clamped);
            eval_cache_valid_word |= eval_cache_bit;
            return clamped;
        };
        if (model_loaded_) {
            // accumulator fast path is HalfKAv2_hm-aware. King moves are
            // detected in apply_piece_delta and trigger acc.valid=false; the
            // refresh below picks up the new king_sq/bucket via the king cache.
            update_phase_for(board_);
            const int idx = std::min(ply, MAX_PLY + 1);
            PendingAccumulator& pending = pending_acc_stack_[idx];
            if (pending.valid && pending.null_move) {
                EvalAccumulator& parent_acc = acc_stack_[pending.parent_acc_idx];
                if (!parent_acc.valid
                    || parent_acc.phase != current_phase_
                    || parent_acc.white_king_sq != board_.king[WHITE]
                    || parent_acc.black_king_sq != board_.king[BLACK]) {
                    refresh_acc_via_king_cache(board_, parent_acc);
                }
                if (parent_acc.valid)
                    {
                        const int value = eval_from_accumulator(parent_acc, board_.stm);
                        return cache_eval(value);
                    }
            }
            materialize_accumulator(idx);
            EvalAccumulator& acc = acc_stack_[idx];
            if (!acc.valid
                || acc.phase != current_phase_
                || acc.white_king_sq != board_.king[WHITE]
                || acc.black_king_sq != board_.king[BLACK]) {
                refresh_acc_via_king_cache(board_, acc);
            }
            if (acc.valid) {
                const int value = eval_from_accumulator(acc, board_.stm);
                return cache_eval(value);
            }
            return cache_eval(0);  // no accumulator (a king is missing): nothing to evaluate
        }
        const int value = mat_eval(board_);
        return cache_eval(value);
    }

    // True when one of `side`'s pieces is attacked by a cheaper enemy piece:
    // pawn x minor/rook/queen, minor x rook/queen, rook x queen. These are the
    // positions where the static eval most often reads too high.
    // Fill stack_[ply]'s threat maps: squares attacked by the side not to
    // move, accumulated from the cheapest attacker class up. Our king is
    // removed from the occupancy so squares behind it along a slider line
    // count as attacked (the king cannot step back along the line).
    void compute_threats(int ply) {
        Stack& s = stack_[ply];
        const int us = board_.stm;
        const int them = us ^ 1;
        const bb::Bitboard occ = board_.occ & ~board_.pieces[us][5];
        bb::Bitboard a = 0;
        for (bb::Bitboard b = board_.pieces[them][0]; b;) a |= bb::PAWN_ATTACKS[them][bb::pop_lsb(b)];
        s.threatPawn = a;
        for (bb::Bitboard b = board_.pieces[them][1]; b;) a |= bb::KNIGHT_ATTACKS[bb::pop_lsb(b)];
        for (bb::Bitboard b = board_.pieces[them][2]; b;) a |= bb::bishop_attacks(bb::pop_lsb(b), occ);
        s.threatMinor = a;
        for (bb::Bitboard b = board_.pieces[them][3]; b;) a |= bb::rook_attacks(bb::pop_lsb(b), occ);
        s.threatRook = a;
        for (bb::Bitboard b = board_.pieces[them][4]; b;) {
            const int sq = bb::pop_lsb(b);
            a |= bb::bishop_attacks(sq, occ) | bb::rook_attacks(sq, occ);
        }
        for (bb::Bitboard b = board_.pieces[them][5]; b;) a |= bb::KING_ATTACKS[bb::pop_lsb(b)];
        s.threatAll = a;
        s.threatHash = ((a >> board_.king[us]) & 1) ? 0 : board_.hash;
    }

#ifdef RDP_VERIFY
    // One reduced-depth-pruning candidate (late quiet, non-PV, rd <= RDP_MAXD), recorded with the move already made.
    struct RdpRec {
        int d, rd, mc, imp, cut, pk, from_att, from_lesser, to_att, to_lesser, thr_hi, thr_hang, rank, castle;
        int hist, gap, fut_margin, lmp_thr;
    };
    void rdpv_rec_fill(RdpRec& r, const Move& m, int ply, int d, int rd, int mc, bool imp, bool cut, int us,
                       int moved_piece, int gap, int fut_margin, int lmp_thr) const {
        const int pk = moved_piece % 6, them = us ^ 1;
        r.d = d; r.rd = rd; r.mc = mc; r.imp = imp; r.cut = cut; r.pk = pk;
        r.castle = m.castle ? 1 : 0;
        r.gap = gap; r.fut_margin = fut_margin; r.lmp_thr = lmp_thr;
        r.hist = quiet_history_score(m, ply, us, moved_piece);
        r.rank = us == WHITE ? (m.to >> 3) : 7 - (m.to >> 3);
        const Stack& st = stack_[ply];  // attacks by `them` before the move
        const std::uint64_t lesser = pk == 1 || pk == 2 ? st.threatPawn : pk == 3 ? st.threatMinor
                                   : pk == 4 ? st.threatRook : 0;
        r.from_att = (int) ((st.threatAll >> m.from) & 1);
        r.from_lesser = (int) ((lesser >> m.from) & 1);
        r.to_att = (int) ((st.threatAll >> m.to) & 1);
        r.to_lesser = (int) ((lesser >> m.to) & 1);
        // Threats the moved piece makes from its new square (board is after the move).
        const bb::Bitboard occ = board_.occ;
        bb::Bitboard att = pk == 0 ? bb::PAWN_ATTACKS[us][m.to] : pk == 1 ? bb::KNIGHT_ATTACKS[m.to]
                         : pk == 2 ? bb::bishop_attacks(m.to, occ) : pk == 3 ? bb::rook_attacks(m.to, occ)
                         : pk == 4 ? (bb::bishop_attacks(m.to, occ) | bb::rook_attacks(m.to, occ))
                         : bb::KING_ATTACKS[m.to];
        bb::Bitboard higher = 0;
        for (int k = pk + 1; k <= 4; ++k) higher |= board_.pieces[them][k];
        if (pk == 0) higher = board_.pieces[them][1] | board_.pieces[them][2] | board_.pieces[them][3]
                            | board_.pieces[them][4];
        r.thr_hi = bb::popcount(att & higher);
        int hang = 0;
        for (bb::Bitboard t = att & board_.color_occ[them] & ~board_.pieces[them][5] & ~higher; t;) {
            const int sq = bb::pop_lsb(t);
            if (!is_square_attacked_bb(board_, sq, them)) ++hang;
        }
        r.thr_hang = hang;
    }
    // Every candidate whose search beat alpha is logged; the rest 1 in 16 (weight 16 in the analysis).
    void rdpv_log(const RdpRec& r, int s, int alpha, int beta) {
        static std::uint64_t n = 0;
        FILE* f = rdpv_file();
        if (!f) return;
        const int fh = s > alpha;
        if (!fh && (n++ & 15)) return;
        fprintf(f, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", r.d, r.rd, r.mc, r.imp, r.cut,
                r.pk, r.from_att, r.from_lesser, r.to_att, r.to_lesser, r.thr_hi, r.thr_hang, r.rank, r.castle,
                r.hist, r.gap, r.fut_margin, r.lmp_thr, fh, s >= beta, std::max(-1000, std::min(1000, s - alpha)));
    }
#endif

    bool lesser_piece_threat(int side) const {
        const int them = side ^ 1;
        const bb::Bitboard ep = board_.pieces[them][0];
        bb::Bitboard targets = board_.pieces[side][1] | board_.pieces[side][2]
                             | board_.pieces[side][3] | board_.pieces[side][4];
        while (targets) {
            const int sq = bb::pop_lsb(targets);
            if (bb::PAWN_ATTACKS[side][sq] & ep) return true;
        }
        bb::Bitboard majors = board_.pieces[side][3] | board_.pieces[side][4];
        while (majors) {
            const int sq = bb::pop_lsb(majors);
            if (bb::KNIGHT_ATTACKS[sq] & board_.pieces[them][1]) return true;
            if (bb::bishop_attacks(sq, board_.occ) & board_.pieces[them][2]) return true;
        }
        bb::Bitboard queens = board_.pieces[side][4];
        while (queens) {
            const int sq = bb::pop_lsb(queens);
            if (bb::rook_attacks(sq, board_.occ) & board_.pieces[them][3]) return true;
        }
        return false;
    }

    bool has_non_pawn_material(int side) {
        return (board_.pieces[side][1] | board_.pieces[side][2]
              | board_.pieces[side][3] | board_.pieces[side][4]) != 0;
    }


    bool apply_uci(const std::string& u) {
        auto l = genlegal(board_, false);
        for (const auto& m : l) {
            if (mv_to_uci(m) == u) {
                Undo ud;
                makemove(board_, m, ud);
                game_history_hashes_.push_back(board_.hash);
                return true;
            }
        }
        return false;
    }

    bool is_repetition_hash(std::uint64_t h, int ply) const {
        // Standard rule. A position that recurs inside the current search line (strictly after the root) is a draw:
        // both sides chose to repeat within lines that were searched. A position from the game so far, the root
        // included, counts only if this would be its third occurrence; one earlier occurrence does not stop the
        // opponent from simply playing something else. h is the position at ply + 1.
        // Only positions since the last irreversible move, with the same side to move, can equal this one. A null
        // move is a ply that the fifty-move counter does not count, so the window is widened by the nulls on the line.
        if (ply + 1 <= MAX_PLY) {
            const int window = board_.halfmove + s2_null_count_;
            if (window < 4) return false;
            const int lo = std::max(1, ply + 1 - window);
            for (int i = ply + 1 - 4; i >= lo; i -= 2)
                if (search_path_hashes_[i] == h) return true;
            if (ply + 1 > window) return false;  // the window does not reach back to the root
            int seen = 0;
            for (std::size_t i = rep_scan_from_; i < game_history_hashes_.size(); ++i)
                if (game_history_hashes_[i] == h && ++seen >= 2) return true;
            return false;
        }
        {
            const int last = std::min(ply, MAX_PLY + 1);
            for (int i = 1; i <= last; ++i)
                if (search_path_hashes_[i] == h) return true;
            int seen = 0;
            for (std::size_t i = rep_scan_from_; i < game_history_hashes_.size(); ++i)
                if (game_history_hashes_[i] == h && ++seen >= 2) return true;
            return false;
        }
        if (std::find(game_history_hashes_.begin() + static_cast<std::ptrdiff_t>(rep_scan_from_),
                      game_history_hashes_.end(), h) != game_history_hashes_.end()) {
            return true;
        }
        const int max_idx = std::min(ply, MAX_PLY + 1);
        for (int i = 0; i <= max_idx; ++i) {
            if (search_path_hashes_[i] == h) {
                return true;
            }
        }
        return false;
    }

    SearchConfig config_{};
    // 3 phase-specific weight sets (opening / middlegame / endgame),
    // selected by Lichess phase definition (count of N+B+R+Q both sides).
