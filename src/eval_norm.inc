// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.
//
// The tables in this file were fitted against the output of Berserk 13's evaluation network (Berserk is
// Copyright (C) 2024 Jay Honnold, GPL-3-or-later).

    // Eval normalisation for the search, fitted against Berserk 13's evaluation: our net's score grows
    // faster than a win-probability-stable scale as pieces come off (about 1.3x at 27+ pieces, 2.5x at <= 8), so
    // fixed pruning margins meant different things in different phases. |raw| -> |normalised| per piece band,
    // piecewise linear, odd-symmetric. Bands: 27+, 21-26, 15-20, 9-14, <= 8 pieces.
    static constexpr int kNormN = 15;
    static constexpr int kNormIn[5][15] = {{0, 75, 129, 168, 202, 234, 270, 314, 374, 417, 485, 548, 679, 869, 1094}, {0, 65, 122, 179, 232, 289, 361, 463, 617, 756, 915, 1052, 1254, 1446, 1583}, {0, 67, 137, 216, 313, 437, 613, 848, 1142, 1302, 1495, 1624, 1802, 1957, 2069}, {0, 26, 59, 115, 217, 380, 638, 944, 1317, 1515, 1695, 1809, 1940, 2117, 2284}, {0, 6, 14, 27, 54, 109, 289, 887, 1403, 1636, 1857, 2001, 2159, 2321, 2419}};
    static constexpr int kNormOut[5][15] = {{0, 72, 114, 136, 157, 177, 199, 223, 254, 279, 312, 347, 417, 559, 710}, {0, 51, 93, 129, 160, 190, 227, 279, 350, 414, 502, 607, 742, 893, 1033}, {0, 45, 88, 128, 174, 231, 305, 415, 576, 698, 866, 999, 1163, 1339, 1475}, {0, 7, 26, 57, 107, 178, 282, 434, 636, 777, 946, 1055, 1190, 1334, 1453}, {0, 2, 5, 10, 19, 44, 125, 339, 613, 748, 890, 990, 1133, 1261, 1401}};
    static constexpr int kNormTailX100[5] = {67, 102, 121, 71, 143};
