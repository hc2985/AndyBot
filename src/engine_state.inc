// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

    static constexpr int N_PHASES = 3;
    static constexpr int PHASE_OPENING = 0;
    static constexpr int PHASE_MIDDLEGAME = 1;
    static constexpr int PHASE_ENDGAME = 2;
    std::array<std::shared_ptr<NetworkWeights>, N_PHASES> phase_weights_{};
    int current_phase_ = -1;       // -1 = no valid current phase yet
    int minor_major_count_ = 0;    // incrementally maintained for O(1) phase lookup
    bool model_loaded_ = false;
    std::string model_path_{};
    std::filesystem::path exe_dir_{};
    std::string exe_stem_{};
    bool ponder_enabled_ = true;
    Board board_{};
    std::vector<std::uint64_t> game_history_hashes_{};
    // First game-history index that can still repeat: positions before the root's
    // last irreversible move differ in material or pawns from every later one.
    std::size_t rep_scan_from_ = 0;
    std::array<std::uint64_t, MAX_PLY + 2> search_path_hashes_{};
    // Move that brought us to position at index i (i.e., the move played
    // from position i-1). Used by counter-move heuristic. Sentinel
    // (from = -1) means "no previous move" (root or after null move).
    std::array<Move, MAX_PLY + 2> search_path_moves_{};
    std::shared_ptr<TTBucket[]> tt_{};
    std::size_t tt_bucket_count_ = 0;
    std::size_t tt_bucket_mask_ = 0;
    int tt_mb_ = kDefaultHashMb;
    // history_[stm][from attacked * 2 + to attacked][from][to]
    std::array<std::array<std::array<std::array<std::int16_t, 64>, 64>, 4>, 2> history_{};
    // capture_history_[attacker][to][to defended by the opponent][victim]
    std::array<std::array<std::array<std::array<std::int16_t, 12>, 2>, 64>, 12> capture_history_{};
    std::array<std::array<int, 2>, MAX_PLY> killers_{};

    // Per-ply search stack.
    // Populated by negamax: currentMove/staticEval/inCheck/ply. Other fields
    // are placeholders for future work (Singular Extensions, LMR statScore,
    // cutoff counts, TT-PV propagation).
    struct Stack {
        Move currentMove{};      // Move being searched at this ply
        Move excludedMove{};     // For Singular Extensions (placeholder)
        int killers[2]{-1, -1};  // Killer move keys (mirrors killers_; killers_ stays primary)
        int ply{0};              // Ply depth from root
        int reduction{0};        // For LMR adjustments (placeholder)
        int statScore{0};        // For LMR/pruning decisions (placeholder)
        int cutoffCnt{0};        // Count of cutoffs at this ply (placeholder)
        int staticEval{0};       // Eval at this position
        int movedPiece{-1};      // Piece identity before currentMove was made
        int dext{0};             // EXT_CAP: double extensions on the path to this node
        int curExt{0};           // EXT_CAP: extension of the move being searched from this node
        bool inCheck{false};     // Hoisted in-check flag
        bool ttPv{false};        // Was this in TT principal-variation? (placeholder)
        // Squares the opponent attacks at this node, by cheapest attacker class
        // (set by compute_threats() before quiet moves are scored).
        std::uint64_t threatPawn{0};   // attacked by a pawn
        std::uint64_t threatMinor{0};  // ... by a pawn, knight or bishop
        std::uint64_t threatRook{0};   // ... by a pawn, minor or rook
        std::uint64_t threatAll{0};    // attacked by anything
        std::uint64_t threatHash{0};   // position the map is exact for (0: none, or the side to move is in check)
    };
    // +10 slack so ply+2 / ply+4 lookups past MAX_PLY are still safe.
    Stack stack_[MAX_PLY + 10]{};
    int root_depth_ = 0;      // EXT_CAP: current iterative-deepening depth

    // counter_moves_[stm][prev.from][prev.to] = move_key of best response
    // observed (i.e., the move that caused beta cutoff). Indexed by the
    // *previous* move (opponent's). 32 KB; pure ordering signal.
    std::array<std::array<std::array<int, 64>, 64>, 2> counter_moves_{};
    // 1-ply continuation history: cont_history_[prev_piece*64 + prev_to,
    // cur_piece*64 + cur_to] = score. Heap-allocated (1.15 MB exceeds
    // Windows' 1 MB default stack). Pure ordering signal.
    static constexpr int CMH_DIM = 12 * 64;  // 768
    std::array<std::vector<std::int16_t>, 4> cont_history_{};
    std::vector<EvalAccumulator> acc_stack_{};
    std::vector<PendingAccumulator> pending_acc_stack_{};
    std::vector<std::uint64_t> eval_cache_keys_{};
    std::vector<std::int16_t> eval_cache_values_{};
    std::vector<std::uint64_t> eval_cache_valid_{};
    std::vector<std::uint64_t> eval_cache_replace_{};
    // Refresh cache: flat layout, index via king_cache_index(persp, phase, king_sq).
    // Size 2 * N_PHASES * 64 = 384 entries, ~2.3 MB per Engine.
    std::vector<KingCacheEntry> king_cache_{};
    std::uint64_t nodes_ = 0;
    std::atomic<bool> hard_deadline_active_{false};
    std::uint64_t node_limit_ = UINT64_MAX;  // "go nodes N"; no limit by default
    int sel_depth_ = 0;                      // deepest ply reached in this search (for "seldepth")
    std::atomic<std::int64_t> hard_deadline_ticks_{INT64_MAX};  // steady_clock ticks
    // Pondering: no time limits until ponderhit, which restarts the time-management clock (tm_epoch_ticks_) and arms
    // the hard limit; time spent pondering is the opponent's.
    std::atomic<bool> pondering_{false};
    std::atomic<std::int64_t> tm_epoch_ticks_{0};
    std::atomic<int> hard_limit_ms_{-1};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool>* shared_stop_flag_ = nullptr;
    // Lazy SMP: Threads - 1 helper searches, kept between moves (their histories stay warm). They share the
    // network weights and the transposition table and stop when helpers_stop_ is set.
    std::vector<std::unique_ptr<Engine>> helpers_;
    std::atomic<bool> helpers_stop_{false};
    // A helper's deepest completed iteration in the current search (read by the main thread after the join).
    int helper_done_depth_ = 0, helper_done_score_ = 0, helper_done_key_ = -1;
    int threads_ = ANDYBOT_DEFAULT_THREADS;
