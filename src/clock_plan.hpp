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

// clock_plan.hpp - AndyBot time management.
//
// Two pure functions, no engine state:
//   plan_move()          how long one move may take: a target (when to stop starting new iterations, before scaling)
//                        and a hard limit (when to abort), from the clock the GUI sends;
//   iteration_factors()  after every finished iteration, three factors that scale the target: how settled the best
//                        move is, how much the score has been falling, and how concentrated the search is on the
//                        best move.
// The model (projected clock over a horizon, three-factor scaling) and its constants are ported from Berserk 13's
// time management (Jay Honnold, GPL-3); the low-clock cap is AndyBot's. See README credits. Kept free of engine
// headers so tools can test it directly.
#pragma once

#include <algorithm>
#include <optional>

namespace andybot::clock_plan {

// ---------------------------------------------------------------------------------------------------------------
// Per-move plan
// ---------------------------------------------------------------------------------------------------------------
struct Clock {
    double remaining_ms = 0;            // our clock
    double increment_ms = 0;
    double overhead_ms = 0;             // GUI/communication reserve per move
    std::optional<int> moves_to_go;     // set for "N moves in T" controls
};

struct Plan {
    int target_ms = 1;
    int limit_ms = 1;
};

// One clock regime: what share of the projected clock a move gets, the cap on that share of the bare clock, and
// how far past the target the hard limit may stretch.
struct Regime {
    double horizon_moves;   // increments (and overheads, for sudden death) counted ahead
    double target_cap;      // target <= target_cap * remaining
    double limit_cap;       // limit <= limit_cap * remaining - overhead
    double stretch;         // limit <= stretch * target
};

inline constexpr double kLimitSafetyMs = 10.0;   // every limit ends this far before the computed point
inline constexpr double kRepeatingHorizon = 2.5; // moves-to-go: the target is planned over mtg / 2.5 moves

inline Plan plan_move(const Clock& c, double sudden_death_share) {
    const double t = c.remaining_ms, inc = c.increment_ms, oh = c.overhead_ms;
    double target, limit;
    if (c.moves_to_go.has_value() && *c.moves_to_go > 0) {
        // Repeating control: the clock refills after mtg moves; plan over a fraction of them.
        const Regime r{static_cast<double>(*c.moves_to_go), 0.9, 0.8, 5.5};
        const double projected = std::max(1.0, t + r.horizon_moves * inc - oh);
        const double spread = std::max(1.0, r.horizon_moves / kRepeatingHorizon);
        target = std::min(t * r.target_cap, 0.9 * projected / spread);
        limit = std::min(t * r.limit_cap - oh, target * r.stretch) - kLimitSafetyMs;
    } else {
        // Sudden death (with or without increment): project the clock over a fixed horizon of future moves.
        const Regime r{50.0, 0.4193, 0.9221, 5.9280};
        const double projected = std::max(1.0, t + r.horizon_moves * inc - r.horizon_moves * oh);
        target = std::min(t * r.target_cap, projected * sudden_death_share);
        limit = std::min(t * r.limit_cap - oh, target * r.stretch) - kLimitSafetyMs;
    }
    // Low clock (AndyBot): the limit never exceeds half of what is left after the overhead and a small reserve.
    const double reserve = std::max(20.0, t / 100.0);
    const double half_usable = std::max(1.0, (t - oh - reserve) / 2.0);
    limit = std::max(1.0, std::min(limit, half_usable));
    target = std::max(1.0, std::min(target, limit));
    return Plan{static_cast<int>(target), static_cast<int>(limit)};
}

// ---------------------------------------------------------------------------------------------------------------
// Scaling after each finished iteration
// ---------------------------------------------------------------------------------------------------------------
struct Signals {
    int settled_iterations = 0;          // consecutive iterations that kept the same best move (0..10)
    int score = 0;                       // this iteration's score
    int score_three_back = 0;            // the score three iterations ago
    std::optional<int> last_move_score;  // the final score of our previous move's search, if any
    double best_move_node_share = 0;     // share of this search's nodes spent under the best move (0..1)
    bool winning_mate = false;           // the score is a mate for us
};

struct Factors {
    double settled = 1, swing = 1, focus = 1;
};

inline constexpr int kMaxSettled = 10;

inline Factors iteration_factors(const Signals& s, double settled_base, double focus_slope) {
    Factors f;
    // Settled best move: start above 1 and shrink with every iteration that keeps it.
    f.settled = settled_base - 0.0533 * s.settled_iterations;

    // Score swing: only drops count. Without a previous move to compare with, the in-search drop counts double.
    double search_drop = s.score_three_back - s.score;
    double move_drop = 0.0;
    if (s.last_move_score.has_value()) move_drop = *s.last_move_score - s.score;
    else search_drop *= 2;
    f.swing = 0.1127 + 0.0262 * search_drop * (search_drop > 0) + 0.0261 * move_drop * (move_drop > 0);
    f.swing = std::max(0.5028, std::min(1.6561, f.swing));

    // Focus: the more of the tree went to moves other than the best one, the longer we keep looking.
    f.focus = std::max(0.5630, (1.0 - s.best_move_node_share) * focus_slope + 0.4499);
    if (s.winning_mate) f.focus = 0.5;
    return f;
}

// Milliseconds after the search start up to which a new iteration may begin.
inline long long next_iteration_deadline_ms(double target_ms, const Factors& f) {
    return static_cast<long long>(target_ms * f.settled * f.swing * f.focus);
}

}  // namespace andybot::clock_plan
