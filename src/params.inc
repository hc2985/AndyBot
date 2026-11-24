// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.
//
// Some of the S2_* default values (listed below) are Berserk 13's tuned parameters, Copyright (C) 2024 Jay Honnold, licensed under the GNU
// General Public License version 3 or later.

// TM_SOFT_PCT: a new iteration starts only before soft budget * TM_SOFT_PCT / 100 elapsed.
#ifndef ANDYBOT_T_TM_SOFT_PCT
#define ANDYBOT_T_TM_SOFT_PCT 80
#endif
// S2_*: tunables of the main search (search.inc). The null-move (NMP_*), futility (FUT_*), quiet-pruning (PR_*),
// singular-extension (SE_*) and history-bonus (HB_*) numbers come from AndyBot's own tuning (Nov 2026); the
// others are Berserk 13's values for the same rules.
#ifndef ANDYBOT_T_S2_RFP_DEPTH
#define ANDYBOT_T_S2_RFP_DEPTH 70
#endif
#ifndef ANDYBOT_T_S2_RFP_IMPROVING
#define ANDYBOT_T_S2_RFP_IMPROVING 118
#endif
#ifndef ANDYBOT_T_S2_RAZOR
#define ANDYBOT_T_S2_RAZOR 214
#endif
#ifndef ANDYBOT_T_S2_NMP_DEPTH_X1024
#define ANDYBOT_T_S2_NMP_DEPTH_X1024 440
#endif
#ifndef ANDYBOT_T_S2_NMP_MARGIN      // null move needs eval >= beta + this
#define ANDYBOT_T_S2_NMP_MARGIN 30
#endif
#ifndef ANDYBOT_T_S2_NMP_EVAL_X1024  // extra reduction per point of eval above beta, x1024 (capped at 4 plies)
#define ANDYBOT_T_S2_NMP_EVAL_X1024 3
#endif
// Numbers of AndyBot's own quiet-pruning, null-move, singular-extension and history-bonus rules.
#ifndef ANDYBOT_T_S2_PR_DEPTH  // quiet pruning applies below this reduced depth
#define ANDYBOT_T_S2_PR_DEPTH 6
#endif
#ifndef ANDYBOT_T_S2_PR_HIST_DIV  // move history counts as history / this points in the reach estimate
#define ANDYBOT_T_S2_PR_HIST_DIV 107
#endif
#ifndef ANDYBOT_T_S2_PR_SEEQ  // a quiet may lose this much material per ply of reduced depth
#define ANDYBOT_T_S2_PR_SEEQ 128
#endif
#ifndef ANDYBOT_T_S2_PR_CAPH_DIV  // capture history / this shifts the material a capture may lose
#define ANDYBOT_T_S2_PR_CAPH_DIV 64
#endif
#ifndef ANDYBOT_T_S2_NMP_R0  // null-move reduction at depth 0, in 1/1024 ply
#define ANDYBOT_T_S2_NMP_R0 4096
#endif
#ifndef ANDYBOT_T_S2_NMP_EVAL_CAP  // most the eval term may add to the null-move reduction, in 1/1024 ply
#define ANDYBOT_T_S2_NMP_EVAL_CAP 4096
#endif
#ifndef ANDYBOT_T_S2_SE_GAIN  // extra extension per point the alternatives fall short, in 1/1024 ply
#define ANDYBOT_T_S2_SE_GAIN 40
#endif
#ifndef ANDYBOT_T_S2_SE_NEG_DIV  // points past the bar that deepen the cut of a non-singular table move by a ply
#define ANDYBOT_T_S2_SE_NEG_DIV 6
#endif
#ifndef ANDYBOT_T_S2_SE_PROBE_X64  // depth of the singular probe as a share of (depth - 1), x64
#define ANDYBOT_T_S2_SE_PROBE_X64 32
#endif
#ifndef ANDYBOT_T_S2_HB_MAX  // largest history bonus
#define ANDYBOT_T_S2_HB_MAX 1800
#endif
#ifndef ANDYBOT_T_S2_HB_LIN  // history bonus per ply of depth
#define ANDYBOT_T_S2_HB_LIN 190
#endif
#ifndef ANDYBOT_T_S2_PROBCUT
#define ANDYBOT_T_S2_PROBCUT 172
#endif
#ifndef ANDYBOT_T_S2_FUT_BASE
#define ANDYBOT_T_S2_FUT_BASE 89
#endif
#ifndef ANDYBOT_T_S2_FUT_SLOPE
#define ANDYBOT_T_S2_FUT_SLOPE 66
#endif
#ifndef ANDYBOT_T_S2_SEEC_X100
#define ANDYBOT_T_S2_SEEC_X100 9406
#endif
#ifndef ANDYBOT_T_S2_SE_BETA_X64
#define ANDYBOT_T_S2_SE_BETA_X64 50
#endif
#ifndef ANDYBOT_T_S2_LMR_HIST
#define ANDYBOT_T_S2_LMR_HIST 8192
#endif
#ifndef ANDYBOT_T_S2_LMR_DIV_X10000
#define ANDYBOT_T_S2_LMR_DIV_X10000 20385
#endif
#ifndef ANDYBOT_T_S2_LMR_BASE_X10000
#define ANDYBOT_T_S2_LMR_BASE_X10000 2429
#endif
#ifndef ANDYBOT_T_S2_LMP_NONIMP_X10000
#define ANDYBOT_T_S2_LMP_NONIMP_X10000 3503
#endif
#ifndef ANDYBOT_T_S2_LMP_IMP_X10000
#define ANDYBOT_T_S2_LMP_IMP_X10000 9911
#endif
#ifndef ANDYBOT_T_S2_LMR_DEEPER
#define ANDYBOT_T_S2_LMR_DEEPER 69
#endif
#ifndef ANDYBOT_T_S2_QS_FUT
#define ANDYBOT_T_S2_QS_FUT 63
#endif
#ifndef ANDYBOT_T_S2_NORM_PCT
#define ANDYBOT_T_S2_NORM_PCT 100
#endif
#ifndef ANDYBOT_T_S2_TM_ALLOC_X10000
#define ANDYBOT_T_S2_TM_ALLOC_X10000 575
#endif
#ifndef ANDYBOT_T_S2_TM_NODE_X10000
#define ANDYBOT_T_S2_TM_NODE_X10000 22669
#endif
#ifndef ANDYBOT_T_S2_TM_STAB_X10000
#define ANDYBOT_T_S2_TM_STAB_X10000 13110
#endif
#define TP(NAME) (ANDYBOT_T_##NAME)
