// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

#pragma once

namespace andybot::search_score {

// Mate scores are relative to the current search ply.  Transposition-table
// entries are position-relative and can be probed at a different ply, so store
// mate scores with the ply removed and restore it on probe.
constexpr int to_tt(int score, int ply, int mate_score, int max_ply) {
    if (score >= mate_score - max_ply) return score + ply;
    if (score <= -mate_score + max_ply) return score - ply;
    return score;
}

constexpr int from_tt(int score, int ply, int mate_score, int max_ply) {
    if (score >= mate_score - max_ply) return score - ply;
    if (score <= -mate_score + max_ply) return score + ply;
    return score;
}

}  // namespace andybot::search_score
