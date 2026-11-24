// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

int main(int argc, char** argv) {
    bb::init_all();  // build attack tables + magic bitboards once at startup
    std::filesystem::path exe_path =
        (argc > 0 && argv && argv[0])
            ? std::filesystem::absolute(std::filesystem::path(argv[0]))
            : (std::filesystem::current_path() / "andybot.exe");
    std::filesystem::path exe_dir = exe_path.parent_path();
    std::string exe_stem = exe_path.stem().string();

    std::filesystem::path explicit_model;
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == "--model") { explicit_model = argv[i + 1]; break; }

    Engine engine(exe_dir, exe_stem);
    // Version shown to GUIs. Development builds override it (-DANDYBOT_UCI_VERSION=1.1-dev).
#ifndef ANDYBOT_UCI_VERSION
#define ANDYBOT_UCI_VERSION 1.0
#endif
#define ANDYBOT_STR2(x) #x
#define ANDYBOT_STR(x) ANDYBOT_STR2(x)
    const std::string uci_engine_name = "AndyBot " ANDYBOT_STR(ANDYBOT_UCI_VERSION);
    if (!engine.load_model(explicit_model)) {
        std::cerr << "AndyBot: no network found. Pass --model <file> or put model.abm next to the executable.\n";
        return 2;
    }

    std::thread ponder_thread;
    bool ponder_active = false;
    SearchResult ponder_result{};
    long long ponder_ms = 0;
    std::atomic<bool> search_output_emitted{false};
    std::atomic<bool> search_finished{false};
    std::atomic<bool> hold_bestmove{false};  // go ponder / go infinite: answer only after ponderhit or stop
    std::atomic<bool> search_started{false};
    std::mutex uci_output_mutex;

    auto emit_search_result = [&]() {
        if (search_output_emitted.exchange(true, std::memory_order_acq_rel)) return;
        std::lock_guard<std::mutex> lock(uci_output_mutex);
        // Totals for the whole search (the last iteration line only covers completed iterations).
        if (ponder_result.bestmove == "0000")  // checkmate or stalemate on the board: nothing to search
            std::cout << "info depth 0 score " << Engine::uci_score(ponder_result.score_cp) << "\n";
        else
        std::cout << "info depth " << ponder_result.depth << " seldepth " << ponder_result.seldepth << " score "
                  << Engine::uci_score(ponder_result.score_cp) << " nodes " << ponder_result.nodes << " nps "
                  << ponder_result.nodes * 1000ULL / static_cast<std::uint64_t>(std::max(1LL, ponder_ms))
                  << " hashfull " << ponder_result.hashfull << " time " << ponder_ms << " pv " << ponder_result.pv << "\n";
        std::cout << "bestmove " << ponder_result.bestmove;
        if (!ponder_result.ponder.empty()) std::cout << " ponder " << ponder_result.ponder;
        std::cout << "\n";
        std::cout.flush();
    };

    auto join_ponder = [&](bool output_bestmove) {
        if (!ponder_active) return;
        while (!search_started.load(std::memory_order_acquire)) std::this_thread::yield();
        engine.request_stop();
        if (ponder_thread.joinable()) ponder_thread.join();
        ponder_active = false;
        if (output_bestmove) emit_search_result();
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto tok = split_ws(line);
        if (tok.empty()) continue;

        const std::string& cmd = tok[0];
        if (ponder_active && cmd != "ponderhit" && cmd != "stop" && cmd != "isready" && cmd != "quit") {
            join_ponder(false);
        }

        if (cmd == "uci") {
            std::cout << "id name " << uci_engine_name << "\n";
            std::cout << "id author Hyeonung Cho\n";
            std::cout << "option name Hash type spin default " << Engine::kDefaultHashMb << " min 1 max " << Engine::kMaxHashMb << "\n";
            std::cout << "option name Threads type spin default " << ANDYBOT_DEFAULT_THREADS << " min 1 max "
                      << Engine::kMaxThreads << "\n";
            std::cout << "option name Ponder type check default false\n";
            std::cout << "option name Move Overhead type spin default "
                      << ANDYBOT_DEFAULT_MOVE_OVERHEAD_MS << " min 0 max 5000\n";
            std::cout << "option name Clear Hash type button\n";
            std::cout << "uciok\n";
            std::cout.flush();
        } else if (cmd == "isready") {
            std::cout << "readyok\n";
            std::cout.flush();
        } else if (cmd == "ucinewgame") {
            engine.new_game();
        } else if (cmd == "setoption") {
            std::string name, value; bool in_name = false, in_value = false;
            for (std::size_t i = 1; i < tok.size(); ++i) {
                if (tok[i] == "name") { in_name = true; in_value = false; continue; }
                if (tok[i] == "value") { in_name = false; in_value = true; continue; }
                if (in_name) { if (!name.empty()) name += ' '; name += tok[i]; }
                else if (in_value) { if (!value.empty()) value += ' '; value += tok[i]; }
            }
            if (!name.empty()) engine.set_option(name, value);
        } else if (cmd == "position") {
            PositionState p = parse_position_cmd(tok);
            engine.set_position(p);
        } else if (cmd == "debug_eval") {
            std::cout << "eval cp " << engine.debug_static_eval() << "\n";
            std::cout.flush();
        } else if (cmd == "bench_eval") {
            std::uint64_t iterations = 100000;
            if (tok.size() >= 2) {
                try {
                    iterations = std::max<std::uint64_t>(1, std::stoull(tok[1]));
                } catch (...) {
                    iterations = 100000;
                }
            }
            const auto result = engine.benchmark_static_eval(iterations);
            const double evals_per_second = result.elapsed_us > 0
                ? static_cast<double>(result.iterations) * 1000000.0 /
                      static_cast<double>(result.elapsed_us)
                : 0.0;
            std::cout << "bench_eval iterations " << result.iterations
                      << " elapsed_us " << result.elapsed_us
                      << " evals_per_second " << std::fixed << std::setprecision(1)
                      << evals_per_second
                      << " checksum " << result.checksum << "\n";
            std::cout.flush();
        } else if (cmd == "divide") {
            // Like perft but prints per-root-move counts, so we can compare
            // against Stockfish's `go perft <d>` output and find where we diverge.
            const int depth = (tok.size() >= 2) ? std::max(1, std::atoi(tok[1].c_str())) : 1;
            Board& b = engine.mutable_board();
            std::vector<Move> moves; moves.reserve(128);
            genpseudo<std::vector<Move>>(b, moves, false);
            const int us = b.stm;
            std::uint64_t total = 0;
            for (const Move& m : moves) {
                Undo u; makemove(b, m, u);
                if (incheck(b, us)) { unmakemove(b, u); continue; }
                const std::uint64_t n = (depth == 1) ? 1 : perft_mailbox(b, depth - 1);
                unmakemove(b, u);
                total += n;
                // UCI move format: from_sq + to_sq + promo_char
                std::string mstr;
                mstr += static_cast<char>('a' + (m.from & 7));
                mstr += static_cast<char>('1' + (m.from >> 3));
                mstr += static_cast<char>('a' + (m.to   & 7));
                mstr += static_cast<char>('1' + (m.to   >> 3));
                if (m.promo) mstr += static_cast<char>(std::tolower(m.promo));
                std::cout << "info string  " << mstr << ": " << n << "\n";
            }
            std::cout << "info string total: " << total << "\n";
            std::cout.flush();
        } else if (cmd == "hashinfo") {
            // Reports TT fill: configured size, total slots, sampled occupancy %.
            // Default samples up to first 4096 buckets (16K slots) for speed;
            // pass `hashinfo full` to scan the entire table.
            const bool full_scan = (tok.size() >= 2 && tok[1] == "full");
            const std::size_t max_b = full_scan ? engine.tt_total_buckets()
                                                : std::min<std::size_t>(4096, engine.tt_total_buckets());
            std::size_t occ = 0, samp = 0;
            engine.tt_sample(max_b, occ, samp);
            const double pct = samp > 0 ? (100.0 * occ / samp) : 0.0;
            const int permille = samp > 0 ? static_cast<int>(1000.0 * occ / samp) : 0;
            std::cout << "info string TT size=" << engine.tt_size_mb() << "MB"
                      << "  total_slots=" << engine.tt_total_slots()
                      << "  sampled=" << samp << (full_scan ? " (full scan)" : " (first 16K)")
                      << "\n";
            std::cout << "info string TT occupied=" << occ << " / " << samp
                      << "  fill=" << std::fixed << std::setprecision(2) << pct << "%"
                      << "  hashfull=" << permille << " (per-mille; UCI standard)\n";
            std::cout.flush();
        } else if (cmd == "perft") {
            // Usage: perft <depth>   — runs both mailbox and bitboard perft on
            // the current position and reports counts + per-second + comparison.
            const int depth = (tok.size() >= 2) ? std::max(1, std::atoi(tok[1].c_str())) : 5;
            Board& b = engine.mutable_board();
            auto t0 = std::chrono::steady_clock::now();
            const std::uint64_t mb_n = perft_mailbox(b, depth);
            auto t1 = std::chrono::steady_clock::now();
            const std::uint64_t bb_n = perft_bb(b, depth);
            auto t2 = std::chrono::steady_clock::now();
            const auto mb_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            const auto bb_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << "info string perft depth " << depth << "\n";
            std::cout << "info string  mailbox: nodes=" << mb_n << "  time=" << mb_ms << "ms"
                      << "  nps=" << (mb_ms > 0 ? mb_n * 1000ULL / mb_ms : 0) << "\n";
            std::cout << "info string  bitboard:nodes=" << bb_n << "  time=" << bb_ms << "ms"
                      << "  nps=" << (bb_ms > 0 ? bb_n * 1000ULL / bb_ms : 0) << "\n";
            std::cout << "info string  match: " << (mb_n == bb_n ? "OK" : "*** MISMATCH ***") << "\n";
            std::cout.flush();
        } else if (cmd == "go") {
            const bool is_ponder_go = std::find(tok.begin(), tok.end(), "ponder") != tok.end();
            const bool is_infinite_go = std::find(tok.begin(), tok.end(), "infinite") != tok.end();
            int depth = MAX_PLY - 1;
            std::optional<int> time_budget_ms = std::nullopt;
            std::optional<int> hard_budget_ms = std::nullopt;
            auto d = parse_go_depth(tok);
            std::optional<std::uint64_t> go_nodes;
            for (std::size_t i = 0; i + 1 < tok.size(); ++i)
                if (tok[i] == "nodes") go_nodes = std::strtoull(tok[i + 1].c_str(), nullptr, 10);
            engine.set_node_limit(go_nodes.value_or(0));
            if (d.has_value()) {
                depth = d.value();
            } else if (go_nodes.has_value() && go_nodes.value() > 0) {
                // Node-limited search: bounded by the node count instead of the clock.
            } else {
                const auto move_time = parse_go_value(tok, "movetime");
                int available_ms = move_time.value_or(0);
                {
                    auto wtime = parse_go_value(tok, "wtime");
                    auto btime = parse_go_value(tok, "btime");
                    auto winc = parse_go_value(tok, "winc");
                    auto binc = parse_go_value(tok, "binc");
                    auto moves_to_go = parse_go_value(tok, "movestogo");

                    std::optional<int> side_time;
                    int increment_ms = 0;
                    if (engine.side_to_move() == WHITE) {
                        if (wtime.has_value()) side_time = wtime.value();
                        else if (btime.has_value()) side_time = btime.value();
                        increment_ms = winc.value_or(0);
                    } else {
                        if (btime.has_value()) side_time = btime.value();
                        else if (wtime.has_value()) side_time = wtime.value();
                        increment_ms = binc.value_or(0);
                    }
                    if (!move_time.has_value()) available_ms = side_time.value_or(0);
                    andybot::time_control::ClockInput clock;
                    if (side_time.has_value()) clock.remaining_ms = std::max(0, side_time.value());
                    clock.increment_ms = increment_ms;
                    if (moves_to_go.has_value() && moves_to_go.value() > 0) clock.moves_to_go = moves_to_go.value();
                    if (move_time.has_value()) clock.move_time_ms = move_time.value();
                    clock.game_ply = engine.game_ply();
                    clock.move_overhead_ms = engine.config().move_overhead_ms;
                    clock.ponder = false;  // budgets are computed as usual; the engine applies them from ponderhit
                    clock.infinite = is_infinite_go;
                    const auto allocation = andybot::time_control::allocate(clock);
                    time_budget_ms = allocation.soft_ms;
                    hard_budget_ms = allocation.hard_ms;
                    // A fixed time per move is a limit to use up, not a clock to manage: no early stop between
                    // iterations, only the hard limit.
                    if (move_time.has_value() && allocation.hard_ms.has_value()) time_budget_ms.reset();
                    if (side_time.has_value() && !move_time.has_value() && !is_infinite_go
                        && allocation.soft_ms.has_value() && allocation.hard_ms.has_value()) {
                        andybot::clock_plan::Clock c;
                        c.remaining_ms = std::max(0, side_time.value());
                        c.increment_ms = increment_ms;
                        c.overhead_ms = engine.config().move_overhead_ms;
                        if (moves_to_go.has_value() && moves_to_go.value() > 0) c.moves_to_go = moves_to_go.value();
                        const auto plan = andybot::clock_plan::plan_move(c, TP(S2_TM_ALLOC_X10000) / 10000.0);
                        time_budget_ms = plan.target_ms;
                        hard_budget_ms = plan.limit_ms;
                    }
                }


                // Low-material endgame depth boost.
                // STRICTLY targets the 3-pieces-on-board case: K + K + 1 piece.
                // That's KQ vs K, KR vs K, KB vs K, KN vs K — the basic mating
                // (or drawn) endgames. Bare 1-piece-each (or 1-piece-one-side)
                // configurations with low branching where deep search reliably
                // finds mate.
                //
                // Strictly NOT for: any position with pawns, any position with
                // 2+ non-king pieces. Those have too much branching for depth
                // 22+ to be worth the time cost.
                // Independent from the sub-1-minute fast-play logic above.
                if (engine.total_non_king_pieces() == 1 && available_ms > 10000) {
                    const int generous_budget = std::max(50, available_ms / 5);
                    if (!time_budget_ms.has_value() || generous_budget > time_budget_ms.value()) {
                        time_budget_ms = generous_budget;
                        hard_budget_ms = std::min(available_ms - engine.config().move_overhead_ms,
                                                  std::max(generous_budget, generous_budget * 2));
                    }
                }
            }

            join_ponder(false);
            engine.set_pondering(is_ponder_go);
            hold_bestmove.store(is_ponder_go || is_infinite_go);
            ponder_active = true;
            ponder_result = SearchResult{};
            ponder_ms = 0;
            search_output_emitted.store(false, std::memory_order_release);
            search_finished.store(false);
            search_started.store(false, std::memory_order_release);
            ponder_thread = std::thread([
                &engine, &ponder_result, &ponder_ms, &emit_search_result, &search_started, &search_finished,
                &hold_bestmove, depth, time_budget_ms, hard_budget_ms
            ]() {
                const auto t0 = std::chrono::steady_clock::now();
                ponder_result = engine.go_depth(
                    depth, time_budget_ms, hard_budget_ms,
                    &search_started
                );
                const auto t1 = std::chrono::steady_clock::now();
                ponder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                search_finished.store(true);
                if (!hold_bestmove.load()) emit_search_result();
            });
        } else if (cmd == "ponderhit") {
            // The expected move was played: keep searching on our own clock. Whoever finishes last answers.
            engine.ponderhit();
            hold_bestmove.store(false);
            if (search_finished.load()) emit_search_result();
        } else if (cmd == "stop") {
            engine.set_pondering(false);
            hold_bestmove.store(false);
            join_ponder(true);
        } else if (cmd == "quit") {
            join_ponder(false);
            break;
        }
    }
    // Input closed without "quit" (a GUI that just drops the pipe): stop and join the search before leaving.
    engine.set_pondering(false);
    join_ponder(false);
    return 0;
}
