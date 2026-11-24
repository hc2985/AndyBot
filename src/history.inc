// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.
//
// Portions of this file are derived from Berserk 13, Copyright (C) 2024 Jay Honnold, licensed under the GNU
// General Public License version 3 or later (table dimensions and update rule); ported to C++ and modified for
// AndyBot in November 2026.

    static constexpr int HISTORY_LIMIT = 16384;
    static constexpr std::array<int, 4> CONTINUATION_LAGS{1, 2, 4, 6};

    template <typename T>
    static void gravity_update(T& entry, int bonus, int limit = HISTORY_LIMIT) {
        const int bounded = std::clamp(bonus, -limit, limit);
        const int old = static_cast<int>(entry);
        const int next = old + bounded - old * std::abs(bounded) / limit;
        entry = static_cast<T>(std::clamp(next, -limit, limit));
    }


    // Quiet history entry for the node at `ply`: the table is split by whether
    // the move leaves an attacked square and whether it lands on one. Uses the
    // node's stored threat map, so it is valid after the move is made too.
    int threat_bucket(int ply, int from, int to) const {
        const std::uint64_t t = stack_[ply].threatAll;
        return static_cast<int>((t >> from) & 1) * 2 + static_cast<int>((t >> to) & 1);
    }
    std::int16_t& quiet_hist(int side, int ply, int from, int to) {
        return history_[side][threat_bucket(ply, from, to)][from][to];
    }
    int quiet_hist(int side, int ply, int from, int to) const {
        return history_[side][threat_bucket(ply, from, to)][from][to];
    }

    int capture_history_score(const Move& move) const {
        const int attacker = piece_index(board_.sq[move.from]);
        const int victim = piece_index(move.ep
            ? (board_.stm == WHITE ? 'p' : 'P')
            : board_.sq[move.to]);
        if (attacker < 0 || victim < 0) return 0;
        return cap_hist(attacker, move.to, victim);
    }
    // Board must be the position before the capture is made.
    int cap_defended(int to) const {
        return is_square_attacked_bb(board_, to, board_.stm ^ 1) ? 1 : 0;
    }
    // Same answer from the node's threat map when it is current.
    int capture_history_score_at(const Move& move, int ply) const {
        const Stack& st = stack_[ply];
        if (st.threatHash == board_.hash && st.threatHash != 0) {
            const int attacker = piece_index(board_.sq[move.from]);
            const int victim = piece_index(move.ep ? (board_.stm == WHITE ? 'p' : 'P') : board_.sq[move.to]);
            if (attacker < 0 || victim < 0) return 0;
            return capture_history_[attacker][move.to][(st.threatAll >> move.to) & 1][victim];
        }
        return capture_history_score(move);
    }
    std::int16_t& cap_hist(int attacker, int to, int victim) {
        return capture_history_[attacker][to][cap_defended(to)][victim];
    }
    int cap_hist(int attacker, int to, int victim) const {
        return capture_history_[attacker][to][cap_defended(to)][victim];
    }
