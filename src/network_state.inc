// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

    using AccumulatorValue = std::int16_t;
    using AccumulatorArray = std::array<AccumulatorValue, EMB_DIM>;

    struct EvalAccumulator {
        bool valid = false;
        int phase = -1;
        int white_king_sq = -1;
        int black_king_sq = -1;
        // HalfKAv2_hm bucket per perspective, computed when king_sq is
        // set in refresh. Used by apply_piece_delta to keep deltas O(1) on
        // non-king moves without recomputing the bucket each call.
        int white_bucket = -1;
        int black_bucket = -1;
        alignas(64) AccumulatorArray us_white{};
        alignas(64) AccumulatorArray us_black{};
    };

    struct PendingAccumulator {
        bool valid = false;
        bool null_move = false;
        bool king_move = false;  // build through the king cache when first needed
        int parent_acc_idx = 0;
        Move move{};
        Undo undo{};
    };

    // Tier-1 refresh cache. One entry per (perspective, phase, king_sq).
    // Each entry stores the piece array used to build us/them, plus the
    // resulting embedding vectors. On refresh, diff stored pieces against
    // the current board and apply only the deltas — replaces the 30+ row
    // scratch sum done by rebuild_acc_from_board.
    struct KingCacheEntry {
        bool valid = false;
        std::array<char, 64> pieces{};
        bb::Bitboard piece_bbs[2][6] = {};
        alignas(64) AccumulatorArray us{};
    };
