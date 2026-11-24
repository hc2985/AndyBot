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

// root_window.hpp - aspiration window for the root loop of the second search.
//
// Each iteration searches the root inside [lo, hi] around an expected score. A result outside the window is a miss;
// the window then moves towards the result and grows, and the root is searched again.
//   - no window for the first iterations or without an expected score;
//   - near mate-scale scores a side is opened completely (mate hunt);
//   - a low miss pulls the top edge halfway down and retries at the full iteration depth;
//   - a high miss raises the top edge and retries one ply shallower (unless the score is a win);
//   - after every miss the half-width grows by 81/64.
// Ported from Berserk 13's aspiration loop (Jay Honnold, GPL-3), constants included; see README credits.
#pragma once

#include <algorithm>
#include <cstdlib>

namespace andybot::root_window {

inline constexpr int kFirstWindowDepth = 5;  // iterations below this depth use the full window
inline constexpr int kStartHalfWidth = 9;
inline constexpr int kMateHuntEdge = 2000;   // an edge beyond +-this opens that side fully

struct Window {
    int lo = 0, hi = 0;   // alpha and beta of the next root search
    int half_width = 0;
    int depth = 0;        // depth of the next root search
};

inline Window open(int iteration, bool have_expected, int expected, int inf) {
    Window w;
    w.depth = iteration;
    if (have_expected && iteration >= kFirstWindowDepth) {
        w.half_width = kStartHalfWidth;
        w.lo = std::max(-inf, expected - w.half_width);
        w.hi = std::min(inf, expected + w.half_width);
    } else {
        w.half_width = inf;
        w.lo = -inf;
        w.hi = inf;
    }
    return w;
}

inline void open_mate_sides(Window& w, int inf) {
    if (w.lo < -kMateHuntEdge) w.lo = -inf;
    if (w.hi > kMateHuntEdge) w.hi = inf;
}

inline void grow(Window& w) { w.half_width = w.half_width * 81 / 64; }

inline void missed_low(Window& w, int iteration, int inf) {
    w.hi = (w.lo + w.hi) / 2;
    w.lo = std::max(-inf, w.lo - w.half_width);
    w.depth = iteration;
    grow(w);
}

inline void missed_high(Window& w, int score, int win_bound, int inf) {
    w.hi = std::min(inf, w.hi + w.half_width);
    if (std::abs(score) < win_bound) --w.depth;
    grow(w);
}

}  // namespace andybot::root_window
