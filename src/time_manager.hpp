// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

namespace andybot::time_control {

struct ClockInput {
    std::optional<std::int64_t> remaining_ms;
    std::optional<std::int64_t> increment_ms;
    std::optional<int> moves_to_go;
    std::optional<std::int64_t> move_time_ms;
    int game_ply = 0;
    int move_overhead_ms = 30;
    bool ponder = false;
    bool infinite = false;
    int soft_scale_pct = 100;   // multiplier on the soft budget, percent
};

struct Allocation {
    std::optional<int> soft_ms;
    std::optional<int> hard_ms;
    std::int64_t available_ms = 0;
    int estimated_moves_to_go = 0;
};

inline Allocation allocate(const ClockInput& in) {
    Allocation out;
    if (in.infinite || in.ponder) return out;

    const std::int64_t overhead = std::max<std::int64_t>(0, in.move_overhead_ms);
    if (in.move_time_ms.has_value()) {
        const std::int64_t raw = std::max<std::int64_t>(1, *in.move_time_ms);
        const int hard = static_cast<int>(std::clamp<std::int64_t>(raw - overhead, 1, INT32_MAX));
        out.available_ms = raw;
        out.soft_ms = hard;
        out.hard_ms = hard;
        out.estimated_moves_to_go = 1;
        return out;
    }
    if (!in.remaining_ms.has_value()) return out;

    // A GUI is allowed to report an exhausted clock as exactly zero. That is
    // still a clocked search, not an unbounded `go`: return a move immediately.
    const std::int64_t remaining = std::max<std::int64_t>(0, *in.remaining_ms);
    const std::int64_t reserve = overhead + std::max<std::int64_t>(20, remaining / 100);
    const std::int64_t usable = std::max<std::int64_t>(1, remaining - reserve);
    const int estimated = in.moves_to_go.has_value() && *in.moves_to_go > 0
        ? std::clamp(*in.moves_to_go, 1, 80)
        : std::clamp(40 - in.game_ply / 2, 12, 30);
    const std::int64_t increment = std::max<std::int64_t>(0, in.increment_ms.value_or(0));
    std::int64_t soft = usable / estimated + (increment * 4) / 5;
    if (in.soft_scale_pct != 100) soft = soft * std::max(1, in.soft_scale_pct) / 100;
    soft = std::clamp<std::int64_t>(soft, 1, usable);
    std::int64_t hard = std::max<std::int64_t>(soft + 50, soft * 3);
    hard = std::clamp<std::int64_t>(hard, soft, usable);
    // Low clock: never plan to spend more than half of what is usable, and keep soft at most half of hard, so the
    // increment rebuilds the bank instead of the engine living on the 50 ms reserve (binds only below ~0.4 s).
    hard = std::max<std::int64_t>(1, std::min(hard, usable / 2));
    soft = std::max<std::int64_t>(1, std::min(soft, std::max<std::int64_t>(1, hard / 2)));

    out.available_ms = remaining;
    out.estimated_moves_to_go = estimated;
    out.soft_ms = static_cast<int>(std::min<std::int64_t>(soft, INT32_MAX));
    out.hard_ms = static_cast<int>(std::min<std::int64_t>(hard, INT32_MAX));
    return out;
}

}  // namespace andybot::time_control
