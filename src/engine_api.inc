// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.
//
// Portions of this file are derived from Berserk 13, Copyright (C) 2024 Jay Honnold, licensed under the GNU
// General Public License version 3 or later (the iteration loop and its stop rule); ported to C++ and modified
// for AndyBot in November 2026.

    explicit Engine(std::filesystem::path exe_dir, std::string exe_stem, bool own_table = true)
        : exe_dir_(std::move(exe_dir)), exe_stem_(std::move(exe_stem)) {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        threads_ = static_cast<int>(
            std::min(static_cast<unsigned>(ANDYBOT_DEFAULT_THREADS), hw));
        acc_stack_.resize(MAX_PLY + 2);
        pending_acc_stack_.resize(MAX_PLY + 2);
        const std::size_t eval_cache_sets = std::size_t{1} << ANDYBOT_EVAL_CACHE_BITS;
        const std::size_t eval_cache_size = eval_cache_sets * ANDYBOT_EVAL_CACHE_WAYS;
        eval_cache_keys_.resize(eval_cache_size);
        eval_cache_values_.resize(eval_cache_size);
        eval_cache_valid_.resize((eval_cache_size + 63) / 64);
        eval_cache_replace_.resize((eval_cache_sets + 63) / 64);
        king_cache_.resize(static_cast<std::size_t>(2) * N_PHASES * 64);
        for (auto& table : cont_history_)
            table.assign(static_cast<std::size_t>(CMH_DIM) * CMH_DIM, 0);
        if (own_table) init_tt(tt_mb_);
        parse_fen(STARTPOS_FEN, board_);
        seed_phase_from_board();
        game_history_hashes_.clear();
        game_history_hashes_.push_back(board_.hash);
        rep_scan_from_ = 0;
        search_path_hashes_.fill(0ULL);
        search_path_hashes_[0] = board_.hash;
        search_path_moves_.fill(Move{});
        for (auto& k : killers_) {
            k[0] = -1;
            k[1] = -1;
        }
        invalidate_eval_accumulators();
    }

    // Network search order: --model (authoritative, for A/B tests), the network built into the executable, then a
    // model.abm next to the executable or in the working directory.
    bool load_model(const std::filesystem::path& explicit_model_path = {}) {
        auto try_load = [this](const std::filesystem::path& p,
                               std::shared_ptr<NetworkWeights>& out) -> bool {
            auto w = std::make_shared<NetworkWeights>();
            if (p.empty() || !std::filesystem::exists(p) || !load_model_file(p, *w)) return false;
            out = std::move(w);
            return true;
        };
        std::array<std::shared_ptr<NetworkWeights>, N_PHASES> loaded;
        std::string loaded_path;

        // Explicit models are authoritative for reproducible A/B matches.
        if (!explicit_model_path.empty()) {
            std::shared_ptr<NetworkWeights> w;
            if (!try_load(explicit_model_path, w)) {
                model_loaded_ = false;
                model_path_.clear();
                for (auto& sp : phase_weights_) sp.reset();
                current_phase_ = -1;
                sync_active_weights_pointer();
                invalidate_eval_accumulators();
                return false;
            }
            for (auto& sp : loaded) sp = w;
            loaded_path = explicit_model_path.string();
        }

        if (!loaded[0]) {
            auto w = std::make_shared<NetworkWeights>();
            if (load_embedded_model(*w)) {
                for (auto& sp : loaded) sp = w;
                loaded_path = "embedded";
            }
        }

        for (const auto& p : {exe_dir_ / "model.abm", std::filesystem::current_path() / "model.abm"}) {
            if (loaded[0]) break;
            std::shared_ptr<NetworkWeights> w;
            if (try_load(p, w)) {
                for (auto& sp : loaded) sp = w;
                loaded_path = p.string();
                break;
            }
        }

        if (!loaded[0]) {
            model_loaded_ = false;
            model_path_.clear();
            for (auto& sp : phase_weights_) sp.reset();
            current_phase_ = -1;
            sync_active_weights_pointer();
            invalidate_eval_accumulators();
            return false;
        }

        for (int ph = 0; ph < N_PHASES; ++ph) phase_weights_[ph] = std::move(loaded[ph]);
        model_loaded_ = true;
        model_path_ = std::move(loaded_path);
        current_phase_ = -1;  // forces phase recompute on next eval
        sync_active_weights_pointer();
        invalidate_eval_accumulators();
        std::fill(eval_cache_valid_.begin(), eval_cache_valid_.end(), 0ULL);
        std::fill(eval_cache_replace_.begin(), eval_cache_replace_.end(), 0ULL);
        return true;
    }

    // UCI option names are case-insensitive. "MoveOverhead" is accepted as well as "Move Overhead".
    void set_option(std::string_view raw_name, std::string_view value) {
        std::string name;
        for (char c : raw_name)
            if (c != ' ') name += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        const int number = std::atoi(std::string(value).c_str());
        const bool on = value == "true" || value == "1";
        if (name == "moveoverhead") config_.move_overhead_ms = std::clamp(number, 0, 5000);
        else if (name == "ponder") ponder_enabled_ = on;
        else if (name == "hash") init_tt(std::clamp(number, 1, kMaxHashMb));
        else if (name == "clearhash") new_game();
        else if (name == "threads") {
            const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
            threads_ = std::clamp(number, 1, std::max(1, std::min(hw, kMaxThreads)));
        }
    }
    static constexpr int kDefaultHashMb = 256;  // CCRL asks for 256 MB and 1 thread as defaults
    static constexpr int kMaxHashMb = 16384;  // the table is the largest power of two that fits
    static constexpr int kMaxThreads = 256;

    Board& mutable_board() { return board_; }  // for perft / debugging only
    int debug_static_eval() { return eval_cp(0); }

    struct EvalBenchmarkResult {
        std::uint64_t iterations = 0;
        std::uint64_t elapsed_us = 0;
        std::int64_t checksum = 0;
    };

    EvalBenchmarkResult benchmark_static_eval(std::uint64_t iterations) {
        for (int i = 0; i < 1024; ++i) (void)eval_cp(0);
        const auto begin = std::chrono::steady_clock::now();
        std::int64_t checksum = 0;
        for (std::uint64_t i = 0; i < iterations; ++i)
            checksum += static_cast<std::int64_t>(eval_cp(0));
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed_us = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        return EvalBenchmarkResult{iterations, elapsed_us, checksum};
    }

    // TT inspection — samples up to N buckets sequentially, returns
    // (occupied_slots, total_slots_sampled). A slot counts as occupied if
    // either of its atomics is nonzero (cleared slots are exactly 0).
    void tt_sample(std::size_t max_buckets, std::size_t& occupied, std::size_t& sampled) const {
        occupied = 0; sampled = 0;
        if (!tt_ || tt_bucket_count_ == 0) return;
        const std::size_t to_scan = std::min<std::size_t>(max_buckets, tt_bucket_count_);
        for (std::size_t b = 0; b < to_scan; ++b) {
            const TTBucket& bucket = tt_[b];
            for (int i = 0; i < TT_BUCKET_N; ++i) {
                const std::uint64_t d   = tt_word_load(bucket.data[i]);
                const std::uint64_t kxd = tt_word_load(bucket.key_xor_data[i]);
                if (d != 0 || kxd != 0) ++occupied;
                ++sampled;
            }
        }
    }
    int tt_size_mb() const { return tt_mb_; }
    std::size_t tt_total_buckets() const { return tt_bucket_count_; }
    std::size_t tt_total_slots() const { return tt_bucket_count_ * TT_BUCKET_N; }

    void set_position(const PositionState& pos) {
        if (!parse_fen(pos.fen, board_)) parse_fen(STARTPOS_FEN, board_);
        game_history_hashes_.clear();
        game_history_hashes_.push_back(board_.hash);
        rep_scan_from_ = 0;
        for (const auto& u : pos.moves) apply_uci(u);
        {
            const std::size_t n = game_history_hashes_.size();
            const std::size_t keep = std::min<std::size_t>(n, static_cast<std::size_t>(board_.halfmove) + 1);
            rep_scan_from_ = n - keep;
        }
        search_path_hashes_.fill(0ULL);
        search_path_hashes_[0] = board_.hash;
        search_path_moves_.fill(Move{});
        invalidate_eval_accumulators();
        seed_phase_from_board();  // re-derive minor_major_count_ after non-incremental change
    }

    void new_game(bool clear_table = true) {
        if (clear_table) clear_tt();
        for (auto& h : helpers_) h->new_game(false);
        history_ = {};
        for (auto& k : killers_) {
            k[0] = -1;
            k[1] = -1;
        }
        for (auto& side_cm : counter_moves_) {
            for (auto& from_cm : side_cm) from_cm.fill(0);
        }
        for (auto& table : cont_history_)
            std::fill(table.begin(), table.end(), static_cast<std::int16_t>(0));
        std::fill(s2_ch_.begin(), s2_ch_.end(), static_cast<std::int16_t>(0));
        std::memset(s2_counters_, 0, sizeof(s2_counters_));
        s2_prev_search_score_ = INT_MIN;
        capture_history_ = {};
        parse_fen(STARTPOS_FEN, board_);
        seed_phase_from_board();
        game_history_hashes_.clear();
        game_history_hashes_.push_back(board_.hash);
        rep_scan_from_ = 0;
        search_path_hashes_.fill(0ULL);
        search_path_hashes_[0] = board_.hash;
        search_path_moves_.fill(Move{});
        invalidate_eval_accumulators();
    }

    SearchResult go_depth(int depth,
                          std::optional<int> soft_budget_ms = std::nullopt,
                          std::optional<int> hard_budget_ms = std::nullopt,
                          std::atomic<bool>* started_signal = nullptr) {
        SearchResult r;
        r.depth = std::max(1, depth);
        nodes_ = 0;
        std::fill(s2_root_avg_.begin(), s2_root_avg_.end(), INT_MIN);
        s2_root_best_key_ = -1;
        s2_tt_age_ = static_cast<std::uint8_t>((s2_tt_age_ + 1) & 7);  // S2F_TTAGE
        stop_requested_.store(false, std::memory_order_release);
        if (started_signal != nullptr) started_signal->store(true, std::memory_order_release);
        sel_depth_ = 0;
        invalidate_eval_accumulators();
        if (model_loaded_) refresh_acc_via_king_cache(board_, acc_stack_[0]);

        const auto search_start = std::chrono::steady_clock::now();
        const bool use_soft_budget = soft_budget_ms.has_value() && soft_budget_ms.value() > 0;
        tm_epoch_ticks_.store(search_start.time_since_epoch().count());
        const int hard_limit = hard_budget_ms.has_value() && hard_budget_ms.value() > 0 ? hard_budget_ms.value() : -1;
        hard_limit_ms_.store(hard_limit);
        hard_deadline_ticks_.store(hard_limit > 0 ? search_start.time_since_epoch().count() + ticks_for_ms(hard_limit)
                                                  : INT64_MAX);
        hard_deadline_active_.store(hard_limit > 0 && !pondering_.load());
        // Milliseconds since the time-management epoch (the search start, or the ponderhit).
        auto since_epoch_ms = [&]() -> long long {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::duration(now_ticks() - tm_epoch_ticks_.load())).count();
        };
        // Soft budget is checked only between iterations: a new iteration starts while elapsed < soft * TM_SOFT_PCT%.
        // A started iteration runs until the hard deadline (checked here between root moves and inside the search),
        // so finished work is no longer thrown away when the soft budget passes mid-iteration.
        // Milliseconds after the epoch up to which a new iteration may start (moved after every finished iteration).
        long long start_until_ms = use_soft_budget
            ? static_cast<long long>(soft_budget_ms.value()) * TP(TM_SOFT_PCT) / 100
            : LLONG_MAX;
        // No soft check before depth 5; from then on the limit is set after every finished iteration.
        if (hard_budget_ms.has_value() && hard_budget_ms.value() > 0) start_until_ms = LLONG_MAX;
        std::array<int, MAX_PLY + 8> s2tm_scores{};
        int s2tm_stab = 0, s2tm_prev_key = -1;
        std::fill(s2_root_nodes_.begin(), s2_root_nodes_.end(), 0);
        auto out_of_time = [&]() -> bool {
            if (stop_requested_.load(std::memory_order_acquire)) return true;
            return hard_deadline_active_.load() && now_ticks() >= hard_deadline_ticks_.load();
        };
        auto may_start_iteration = [&]() -> bool {
            return !out_of_time() && (pondering_.load() || since_epoch_ms() < start_until_ms);
        };

        auto legal = genlegal(board_, false);
        if (legal.empty()) {
            r.bestmove = "0000";
            r.score_cp = incheck(board_, board_.stm) ? -MATE : 0;
            r.nodes = nodes_;
            return r;
        }

        Move best_move = legal.front();
        int best_score = -INF;
        int last_score = 0;
        bool have_last = false;
        int completed_depth = 0;
        eval_cp(0);  // builds the root accumulators and fills the eval cache before the first iteration
        search_path_hashes_.fill(0ULL);
        search_path_hashes_[0] = board_.hash;
        search_path_moves_.fill(Move{});

        // Reset the per-ply stack between iterative-deepening iterations so
        // staleness from a previous (deeper) iteration cannot leak into
        // improving/razoring decisions of the next.
        for (auto& s : stack_) s = Stack{};
        stack_[0].inCheck = incheck(board_, board_.stm);
        stack_[0].staticEval = stack_[0].inCheck ? S2_NO_EVAL : eval_norm(eval_cp_unscaled(0));

        std::vector<std::thread> helper_threads;
        if (threads_ > 1) {
            sync_helpers();
            helpers_stop_.store(false);
            for (std::size_t i = 0; i < helpers_.size(); ++i)
                helper_threads.emplace_back([this, i, max_depth = r.depth]() {
                    helpers_[i]->helper_search(max_depth, static_cast<int>(i));
                });
        }
        for (int iter_depth = 1; iter_depth <= r.depth; ++iter_depth) {
            root_depth_ = iter_depth;
            // Depth 1 always completes (well under a millisecond): never answer with an unsearched move.
            hard_deadline_active_.store(hard_limit_ms_.load() > 0 && !pondering_.load() && iter_depth > 1);
            if (iter_depth > 1 && !may_start_iteration()) break;

            // Root window (root_window.hpp), centred on the expected score: the best move's running average when kept.
            int expected = last_score;
            {
                const int bk = move_key(best_move);
                if (have_last && bk >= 0 && bk <= S2_ROOT_KEYS && s2_root_avg_[static_cast<std::size_t>(bk)] != INT_MIN)
                    expected = s2_root_avg_[static_cast<std::size_t>(bk)];
            }
            andybot::root_window::Window rw = andybot::root_window::open(iter_depth, have_last, expected, INF);
            bool aspiration_retry = false;

            while (true) {
                if (out_of_time()) break;
                andybot::root_window::open_mate_sides(rw, INF);
                const int alpha = rw.lo;
                const int beta = rw.hi;
                const bool aspiration_window =
                    have_last && (alpha > -INF + 1 || beta < INF - 1);

                std::vector<Move> root_moves;
                root_moves.reserve(128);
                genpseudo(board_, root_moves, false);
                int local_best = -INF;
                Move local_best_move = root_moves.front();
                int local_best_key = move_key(local_best_move);
                int move_count = 0;
                int legal_count = 0;
                bool timed_out_iteration = false;

                // The search handles the root itself (reductions and pruning apply there too). It reports its best
                // move through s2_root_best_key_; a fail-low leaves that unset and keeps the previous move.
                {
                    // First root move: the best one so far (a move that just failed high, else the previous best).
                    s2_root_first_key_ = s2_root_best_key_ >= 0 ? s2_root_best_key_ : have_last ? move_key(best_move) : -1;
                    s2_root_best_key_ = -1;
                    const int root_score = search_node(std::max(1, rw.depth), alpha, beta, 0, false);
                    legal_count = 1;
                    move_count = 1;
                    local_best = root_score;
                    local_best_move = best_move;
                    if (s2_root_best_key_ >= 0)
                        for (const auto& rm : root_moves)
                            if (move_key(rm) == s2_root_best_key_) { local_best_move = rm; break; }
                    local_best_key = move_key(local_best_move);
                    if (stop_requested_.load(std::memory_order_acquire)) {
                        timed_out_iteration = true;
                        if (have_last && s2_root_best_key_ >= 0) {  // a completed root move beat the previous best
                            best_move = local_best_move;
                            best_score = s2_root_best_score_;
                        }
                    }
                }

                if (timed_out_iteration) break;
                if (legal_count == 0) {
                    local_best = incheck(board_, board_.stm) ? -MATE : 0;
                    local_best_key = -1;
                }

                if (have_last && local_best <= alpha && alpha > -INF + 1) {
                    aspiration_retry = true;
                    andybot::root_window::missed_low(rw, iter_depth, INF);
                    continue;
                }
                if (have_last && local_best >= beta && beta < INF - 1) {
                    // Search the move that failed high first in the wider window (the table still named the old move).
                    if (local_best_key >= 0) {
                        TTEntry fail_high;
                        fail_high.depth = iter_depth;
                        fail_high.score = local_best;
                        fail_high.best_key = local_best_key;
                        fail_high.bound = TT_LOWER;
                        tt_store(board_.hash, fail_high, 0);
                    }
                    aspiration_retry = true;
                    andybot::root_window::missed_high(rw, local_best, S2_WIN, INF);
                    continue;
                }

                if (legal_count > 0) {
                    const int key = move_key(local_best_move);
                    if (use_soft_budget && iter_depth >= 5 && hard_budget_ms.has_value() && hard_budget_ms.value() > 0) {
                        s2tm_stab = key == s2tm_prev_key ? std::min(andybot::clock_plan::kMaxSettled, s2tm_stab + 1) : 0;
                        andybot::clock_plan::Signals sig;
                        sig.settled_iterations = s2tm_stab;
                        sig.score = local_best;
                        sig.score_three_back = s2tm_scores[static_cast<std::size_t>(iter_depth - 3)];
                        if (s2_prev_search_score_ != INT_MIN) sig.last_move_score = s2_prev_search_score_;
                        const double total_nodes = static_cast<double>(static_cast<std::uint64_t>(nodes_));
                        const double best_nodes = (key >= 0 && key <= S2_ROOT_KEYS)
                            ? static_cast<double>(s2_root_nodes_[static_cast<std::size_t>(key)]) : 0.0;
                        sig.best_move_node_share = total_nodes > 0 ? best_nodes / total_nodes : 0.0;
                        sig.winning_mate = local_best >= S2_WIN;
                        const auto f = andybot::clock_plan::iteration_factors(
                            sig, TP(S2_TM_STAB_X10000) / 10000.0, TP(S2_TM_NODE_X10000) / 10000.0);
                        start_until_ms = andybot::clock_plan::next_iteration_deadline_ms(soft_budget_ms.value(), f);
                    }
                    s2tm_prev_key = key;
                    if (iter_depth >= 0 && iter_depth < MAX_PLY + 8) s2tm_scores[static_cast<std::size_t>(iter_depth)] = local_best;
                }
                best_score = local_best;
                best_move = local_best_move;
                last_score = local_best;
                have_last = true;

                TTEntry e;
                e.depth = iter_depth;
                e.score = local_best;
                e.best_key = local_best_key;
                e.bound = TT_EXACT;
                tt_store(board_.hash, e, 0);
                completed_depth = iter_depth;
                // UCI progress: one line per completed iteration.
                r.pv = principal_variation(best_move, iter_depth);
                r.seldepth = std::max(iter_depth, sel_depth_);
                r.hashfull = hashfull_permille();
                {
                    const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - search_start).count();
                    const std::uint64_t total = search_nodes_so_far();
                    std::cout << "info depth " << iter_depth << " seldepth " << r.seldepth << " score "
                              << uci_score(best_score) << " nodes " << total << " nps "
                              << total * 1000ULL / static_cast<std::uint64_t>(std::max(1LL, ms)) << " hashfull "
                              << r.hashfull << " time " << ms << " pv " << r.pv << std::endl;
                }
                break;
            }

            if (!may_start_iteration()) break;

            // Early-out on found mate. If we've found a forced mate at this
            // iteration depth, no point spending more time searching deeper —
            // a mate is a mate. Save the clock for future moves. The
            // (MATE - MAX_PLY) threshold is the standard "is this a mate score
            // and not just a huge eval" check used by most chess engines.
            if (have_last && std::abs(last_score) >= MATE - MAX_PLY) {
                break;
            }
        }

        if (!helper_threads.empty()) {
            helpers_stop_.store(true);
            for (auto& t : helper_threads) t.join();
            for (auto& h : helpers_) nodes_ += h->nodes_;
            // Play the deepest completed result (ties: the higher score), if its move is legal here.
            int pick_depth = completed_depth, pick_score = best_score, pick_key = -1;
            for (auto& h : helpers_) {
                const bool deeper = h->helper_done_depth_ > pick_depth;
                const bool better_tie = h->helper_done_depth_ == pick_depth && h->helper_done_score_ > pick_score;
                if (h->helper_done_key_ >= 0 && (deeper || better_tie)) {
                    pick_depth = h->helper_done_depth_;
                    pick_score = h->helper_done_score_;
                    pick_key = h->helper_done_key_;
                }
            }
            if (pick_key >= 0 && have_last) {
                for (const Move& m : genlegal(board_, false)) {
                    if (move_key(m) != pick_key) continue;
                    best_move = m;
                    best_score = pick_score;
                    completed_depth = pick_depth;
                    break;
                }
            }
        }
        r.depth = std::max(1, completed_depth);
        r.bestmove = mv_to_uci(best_move);
        if (have_last) s2_prev_search_score_ = best_score;
        if (!have_last) best_score = eval_cp(0);
        r.score_cp = best_score;
        r.nodes = nodes_;
        r.pv = principal_variation(best_move, std::max(1, completed_depth));
        {
            const auto space = r.pv.find(' ');
            if (space != std::string::npos) r.ponder = r.pv.substr(space + 1, r.pv.find(' ', space + 1) - space - 1);
        }
        r.seldepth = std::max(r.depth, sel_depth_);
        r.hashfull = hashfull_permille();
        hard_deadline_active_.store(false);
        return r;
    }

    // "cp N", or "mate N" (moves, negative when we are being mated) for forced mates.
    static std::string uci_score(int score) {
        if (score >= MATE - MAX_PLY) return "mate " + std::to_string((MATE - score + 1) / 2);
        if (score <= -MATE) return "mate 0";
        if (score <= -(MATE - MAX_PLY)) return "mate -" + std::to_string((MATE + score + 1) / 2);
        return "cp " + std::to_string(score);
    }
    // Nodes of all threads. Helpers' counters are read while they run: a slightly stale value is fine for display.
    std::uint64_t search_nodes_so_far() const {
        std::uint64_t total = nodes_;
        for (const auto& h : helpers_) total += h->nodes_;
        return total;
    }
    int hashfull_permille() const {
        std::size_t occupied = 0, sampled = 0;
        tt_sample(250, occupied, sampled);
        return sampled > 0 ? static_cast<int>(occupied * 1000 / sampled) : 0;
    }
    void set_node_limit(std::uint64_t n) { node_limit_ = n > 0 ? n : UINT64_MAX; }

    const SearchConfig& config() const { return config_; }
    bool model_loaded() const { return model_loaded_; }
    int side_to_move() const { return board_.stm; }
    int game_ply() const { return std::max(0, static_cast<int>(game_history_hashes_.size()) - 1); }
    // Total non-king pieces (pawns + minor + major, both sides).
    // Computed on demand from board (not incrementally tracked) — used only
    // for time-budget decisions which happen once per move, not per eval.
    int total_non_king_pieces() const {
        int count = 0;
        for (int sq = 0; sq < 64; ++sq) {
            const char p = board_.sq[sq];
            if (p != '.' && p != 'K' && p != 'k') ++count;
        }
        return count;
    }
    bool ponder_enabled() const { return ponder_enabled_; }
    // --- Lazy SMP -------------------------------------------------------------------------------------------
    // Bring the helper engines to Threads - 1 and to the current position, network and table.
    void sync_helpers() {
        const int want = std::max(0, threads_ - 1);
        if (static_cast<int>(helpers_.size()) != want) {
            helpers_.clear();
            for (int i = 0; i < want; ++i) helpers_.push_back(std::make_unique<Engine>(exe_dir_, exe_stem_, false));
        }
        (void)s2_tables();  // build lazily-built search tables here, before any helper reads them
        for (auto& h : helpers_) {
            Engine& w = *h;
            w.config_ = config_;
            w.model_loaded_ = model_loaded_;
            w.model_path_ = model_path_;
            w.phase_weights_ = phase_weights_;
            w.current_phase_ = current_phase_;
            w.sync_active_weights_pointer();
            w.threads_ = 1;
            w.shared_stop_flag_ = &helpers_stop_;
            w.tt_ = tt_;
            w.tt_bucket_count_ = tt_bucket_count_;
            w.tt_bucket_mask_ = tt_bucket_mask_;
            w.tt_mb_ = tt_mb_;
            w.s2_tt_age_ = s2_tt_age_;
            w.board_ = board_;
            w.game_history_hashes_ = game_history_hashes_;
            w.rep_scan_from_ = rep_scan_from_;
        }
    }
    // A helper's search: iterative deepening on the shared table until helpers_stop_. Odd helpers run one ply
    // ahead of the others so the threads fill the table at different depths. No time management, no output:
    // the main search decides when to stop and which move to play.
    void helper_search(int max_depth, int index) {
        nodes_ = 0;
        std::fill(s2_root_avg_.begin(), s2_root_avg_.end(), INT_MIN);
        std::fill(s2_root_nodes_.begin(), s2_root_nodes_.end(), 0);
        s2_root_best_key_ = -1;
        s2_root_first_key_ = -1;
        stop_requested_.store(false);
        hard_deadline_active_.store(false);
        invalidate_eval_accumulators();
        if (model_loaded_) refresh_acc_via_king_cache(board_, acc_stack_[0]);
        search_path_hashes_.fill(0ULL);
        search_path_hashes_[0] = board_.hash;
        search_path_moves_.fill(Move{});
        for (auto& s : stack_) s = Stack{};
        stack_[0].inCheck = incheck(board_, board_.stm);
        stack_[0].staticEval = stack_[0].inCheck ? S2_NO_EVAL : eval_norm(eval_cp_unscaled(0));
        helper_done_depth_ = 0;
        helper_done_key_ = -1;
        auto halted = [&]() { return helpers_stop_or_none(); };
        int last_score = 0, best_key = -1;
        bool have_last = false;
        // Helper i searches depth d only when (d + i) / s is even, s = 1 + i / 2: the helpers interleave across
        // depths, so at any moment they fill the table at different depths from each other and from the main thread.
        const int block = 1 + index / 2;
        for (int depth = 1; depth <= max_depth; ++depth) {
            if (((depth + index) / block) % 2 != 0) continue;
            root_depth_ = depth;
            int expected = last_score;
            if (have_last && best_key >= 0 && best_key <= S2_ROOT_KEYS
                && s2_root_avg_[static_cast<std::size_t>(best_key)] != INT_MIN)
                expected = s2_root_avg_[static_cast<std::size_t>(best_key)];
            andybot::root_window::Window rw = andybot::root_window::open(depth, have_last, expected, INF);
            while (true) {
                if (halted()) return;
                andybot::root_window::open_mate_sides(rw, INF);
                s2_root_first_key_ = s2_root_best_key_ >= 0 ? s2_root_best_key_ : best_key;
                s2_root_best_key_ = -1;
                const int score = search_node(std::max(1, rw.depth), rw.lo, rw.hi, 0, false);
                if (halted()) return;
                if (s2_root_best_key_ >= 0) best_key = s2_root_best_key_;
                if (have_last && score <= rw.lo && rw.lo > -INF + 1) {
                    andybot::root_window::missed_low(rw, depth, INF);
                    continue;
                }
                if (have_last && score >= rw.hi && rw.hi < INF - 1) {
                    andybot::root_window::missed_high(rw, score, S2_WIN, INF);
                    continue;
                }
                last_score = score;
                have_last = true;
                helper_done_depth_ = depth;
                helper_done_score_ = score;
                helper_done_key_ = best_key;
                break;
            }
        }
    }
    bool helpers_stop_or_none() const {
        return shared_stop_flag_ != nullptr && shared_stop_flag_->load(std::memory_order_acquire);
    }
    // Pondering control. While pondering the search has no time limits; ponderhit makes the opponent's expected
    // move real: the time-management clock restarts now and the hard limit is armed.
    void set_pondering(bool on) { pondering_.store(on); }
    bool pondering() const { return pondering_.load(); }
    void ponderhit() {
        const std::int64_t now = now_ticks();
        tm_epoch_ticks_.store(now);
        const int hard = hard_limit_ms_.load();
        if (hard > 0) {
            hard_deadline_ticks_.store(now + ticks_for_ms(hard));
            hard_deadline_active_.store(true);
        }
        pondering_.store(false);
    }
    // Expected reply to `best` for "bestmove ... ponder ...": the table's move in the position after it, if legal
    // there. Read-only (no table state changes).
    // The principal variation for display and for the ponder move: the best move, then the table's best moves for as
    // long as the table has the position and its move is legal there. Read-only: works on a copy of the board.
    std::string principal_variation(const Move& best, int max_len) const {
        if (best.from < 0) return "0000";
        std::string pv = mv_to_uci(best);
        if (!tt_) return pv;
        Board b = board_;
        Undo u;
        ::makemove(b, best, u);
        std::vector<std::uint64_t> seen{board_.hash};
        for (int len = 1; len < std::min(max_len, MAX_PLY); ++len) {
            if (std::find(seen.begin(), seen.end(), b.hash) != seen.end()) break;  // the line repeats
            seen.push_back(b.hash);
            const TTBucket& bucket = tt_[static_cast<std::size_t>(b.hash) & tt_bucket_mask_];
            int key = -1;
            for (int i = 0; i < TT_BUCKET_N && key < 0; ++i) {
                const std::uint64_t d = tt_word_load(bucket.data[i]);
                if (d == 0 || (tt_word_load(bucket.key_xor_data[i]) ^ d) != b.hash) continue;
                TTEntry e;
                tt_unpack(d, e);
                if (e.best_key != 0xFFFF) key = e.best_key;
                break;
            }
            Move m;
            if (key < 0 || !decode_tt_move(b, key, false, m)) break;
            bool legal = false;
            for (const Move& lm : genlegal(b, false))
                if (move_key(lm) == move_key(m)) { m = lm; legal = true; break; }
            if (!legal) break;
            pv += ' ';
            pv += mv_to_uci(m);
            ::makemove(b, m, u);
        }
        return pv;
    }
    void request_stop() {
        stop_requested_.store(true, std::memory_order_release);
        if (shared_stop_flag_ != nullptr) {
            shared_stop_flag_->store(true, std::memory_order_release);
        }
    }
