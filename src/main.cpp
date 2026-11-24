// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

// main.cpp - the whole engine is this one translation unit; it includes the other files in this directory.
//   board.inc, movegen.inc          position, attacks, move generation
//   network*.inc, eval*.inc         NNUE evaluation (HalfKAv2_hm features, incremental accumulators)
//   search.inc, history.inc, tt.inc alpha-beta search, move ordering, hash table
//   params.inc                      the search's numbers
//   clock_plan.hpp, root_window.hpp time management and aspiration windows
//   engine_*.inc, uci.inc           engine object, threads, UCI protocol
// Build from the repository root, for example:
//   clang++ -std=c++17 -O3 -flto -ffp-contract=off -mavx2 -mfma -mbmi -mbmi2 -mpopcnt -mlzcnt -mf16c src/main.cpp -o andybot

// One switch for the CPU-specific code paths: the AVX-512 build (F, BW, DQ, VL, VNNI) or the AVX2 build.
// These are the two tested configurations; every kernel below picks its variant from this.
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__) && defined(__AVX512VNNI__)
#define ANDYBOT_AVX512 1
#else
#define ANDYBOT_AVX512 0
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <streambuf>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

// The network file is linked into the executable. The path is relative to the directory the compiler runs in
// (the repository root); override it with -DANDYBOT_NET_PATH="\"path\"".
#ifndef ANDYBOT_NET_PATH
#define ANDYBOT_NET_PATH "networks/andybot-1.0.nnue"
#endif
__asm__(
#if defined(_WIN32)
    ".section .rdata,\"dr\"\n"
#else
    ".section .rodata\n"
#endif
    ".balign 64\n"
    ".globl andybot_net_begin\n"
    "andybot_net_begin:\n"
    ".incbin \"" ANDYBOT_NET_PATH "\"\n"
    ".globl andybot_net_end\n"
    "andybot_net_end:\n"
    ".byte 0\n"
    ".text\n");
extern "C" {
extern const unsigned char andybot_net_begin[];
extern const unsigned char andybot_net_end[];
}

#include "search_score.hpp"
#include "time_manager.hpp"
#include "clock_plan.hpp"
#include "root_window.hpp"

namespace {

#ifndef ANDYBOT_DENSE_WIDTH
#define ANDYBOT_DENSE_WIDTH 48
#endif
#ifndef ANDYBOT_DEFAULT_MOVE_OVERHEAD_MS
#define ANDYBOT_DEFAULT_MOVE_OVERHEAD_MS 30
#endif

// Release builds intended for single-CPU tournaments must not silently start
// extra search workers when the GUI leaves Threads at its default.  Existing
// lab/Lichess targets retain their historical default unless they override it.
#ifndef ANDYBOT_DEFAULT_THREADS
#define ANDYBOT_DEFAULT_THREADS 1
#endif

constexpr std::uint32_t N_FEATURES = 24576;    // HalfKAv2_hm = 64*12*64/2 (SF nnue-pytorch DLL flavor, NUM_PT=12)
constexpr std::uint32_t EMB_DIM = 1024;        // HalfKAv2_hm encodes more per feature
constexpr std::uint32_t L1_IN = 1024;         // pairwise mul halves EMB_DIM (1024→512/side), concat = 1024
constexpr std::uint32_t PAIRWISE_HALF = EMB_DIM / 2;   // 512 per side after pairwise mul
constexpr std::uint32_t L1_OUT = ANDYBOT_DENSE_WIDTH;
static_assert(L1_OUT >= 8 && L1_OUT % 8 == 0,
              "dense width must be a positive multiple of eight");
constexpr int REPETITION_CONTEMPT_MAX_CP = 120;
constexpr int REPETITION_CONTEMPT_DIVISOR = 8;

#ifndef ANDYBOT_EVAL_CACHE_BITS
#define ANDYBOT_EVAL_CACHE_BITS 16
#endif
#ifndef ANDYBOT_EVAL_CACHE_WAYS
#define ANDYBOT_EVAL_CACHE_WAYS 2
#endif

// =============================================================================
// BITBOARD INFRASTRUCTURE
// Square layout: bit i represents the square at file (i%8), rank (i/8).
//   a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63   (matches mailbox sq indexing).
// Color codes: 0 = WHITE, 1 = BLACK (matches existing engine constants).
// Piece kind codes (bb-side, distinct from the mailbox char encoding):
//   0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
// =============================================================================
namespace bb {

using Bitboard = std::uint64_t;

// File / rank masks
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
constexpr Bitboard FILE_C_BB = FILE_A_BB << 2;
constexpr Bitboard FILE_D_BB = FILE_A_BB << 3;
constexpr Bitboard FILE_E_BB = FILE_A_BB << 4;
constexpr Bitboard FILE_F_BB = FILE_A_BB << 5;
constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;
constexpr Bitboard NOT_FILE_A = ~FILE_A_BB;
constexpr Bitboard NOT_FILE_H = ~FILE_H_BB;
constexpr Bitboard NOT_FILE_AB = ~(FILE_A_BB | FILE_B_BB);
constexpr Bitboard NOT_FILE_GH = ~(FILE_G_BB | FILE_H_BB);

constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;

// Bit utilities (compiler intrinsics for speed)
inline int lsb(Bitboard b) {
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(b);
#endif
}

inline int popcount(Bitboard b) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(b));
#else
    return __builtin_popcountll(b);
#endif
}

inline int pop_lsb(Bitboard& b) {
    const int s = lsb(b);
    b &= b - 1;
    return s;
}

constexpr Bitboard sq_bb(int sq) { return 1ULL << sq; }

// Precomputed attacks for non-sliding pieces (filled by init_bb()).
Bitboard KNIGHT_ATTACKS[64];
Bitboard KING_ATTACKS[64];
Bitboard PAWN_ATTACKS[2][64];   // [color][sq] — squares the pawn ATTACKS (not pushes)

void init_knight_attacks() {
    static constexpr int off[8][2] = {
        {-2,-1},{-2,+1},{-1,-2},{-1,+2},{+1,-2},{+1,+2},{+2,-1},{+2,+1}
    };
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard a = 0;
        const int r = sq >> 3, f = sq & 7;
        for (auto& o : off) {
            const int nr = r + o[0], nf = f + o[1];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) a |= sq_bb(nr * 8 + nf);
        }
        KNIGHT_ATTACKS[sq] = a;
    }
}

void init_king_attacks() {
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard a = 0;
        const int r = sq >> 3, f = sq & 7;
        for (int dr = -1; dr <= 1; ++dr)
            for (int df = -1; df <= 1; ++df) {
                if (dr == 0 && df == 0) continue;
                const int nr = r + dr, nf = f + df;
                if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) a |= sq_bb(nr * 8 + nf);
            }
        KING_ATTACKS[sq] = a;
    }
}

void init_pawn_attacks() {
    for (int sq = 0; sq < 64; ++sq) {
        const int r = sq >> 3, f = sq & 7;
        Bitboard wa = 0, ba = 0;
        if (r < 7) {
            if (f > 0) wa |= sq_bb(sq + 7);
            if (f < 7) wa |= sq_bb(sq + 9);
        }
        if (r > 0) {
            if (f > 0) ba |= sq_bb(sq - 9);
            if (f < 7) ba |= sq_bb(sq - 7);
        }
        PAWN_ATTACKS[0][sq] = wa;
        PAWN_ATTACKS[1][sq] = ba;
    }
}

// =============================================================================
// MAGIC BITBOARDS for sliding pieces (bishop, rook).
// Magic numbers from the public-domain Pradyumna Kannan / Tord Romstad set,
// widely verified across open-source engines (Stockfish, etc.).
// =============================================================================

struct Magic {
    Bitboard mask;        // relevant occupancy bits for this square
    Bitboard magic;       // multiplier for the hash
    int      shift;       // 64 - popcount(mask)
    Bitboard* attacks;    // pointer into BISHOP_TABLE / ROOK_TABLE
};

Magic BISHOP_MAGICS[64];
Magic ROOK_MAGICS[64];
// Sized large enough for any magic-num set (bishop ~5KB, rook ~800KB).
Bitboard BISHOP_TABLE[5248];
Bitboard ROOK_TABLE[102400];

// Magic numbers - verified from the public-domain Stockfish-compatible set.
constexpr Bitboard ROOK_MAGIC_NUMS[64] = {
    0xa8002c000108020ULL,  0x6c00049b0002001ULL,  0x100200010090040ULL, 0x2480041000800801ULL,
    0x280028004000800ULL,  0x900410008040022ULL,  0x280020001001080ULL, 0x2880002041000080ULL,
    0xa000800080400034ULL, 0x4808020004000ULL,    0x2290802004801000ULL,0x411000d00100020ULL,
    0x402800800040080ULL,  0xb000401004208ULL,    0x2409000100040200ULL,0x1002100004082ULL,
    0x22878001e24000ULL,   0x1090810021004010ULL, 0x801030040200012ULL, 0x500808008001000ULL,
    0xa08018014000880ULL,  0x8000808004000200ULL, 0x201008080010200ULL, 0x801020000441091ULL,
    0x800080204005ULL,     0x1040200040100048ULL, 0x120200402082ULL,    0xd14880480100080ULL,
    0x12040280080080ULL,   0x100040080020080ULL,  0x9020010080800200ULL,0x813241200148449ULL,
    0x491604001800080ULL,  0x100401000402001ULL,  0x4820010021001040ULL,0x400402202000812ULL,
    0x209009005000802ULL,  0x810800601800400ULL,  0x4301083214000150ULL,0x204026458e001401ULL,
    0x40204000808000ULL,   0x8001008040010020ULL, 0x8410820820420010ULL,0x1003001000090020ULL,
    0x804040008008080ULL,  0x12000810020004ULL,   0x1000100200040208ULL,0x430000a044020001ULL,
    0x280009023410300ULL,  0xe0100040002240ULL,   0x200100401700ULL,    0x2244100408008080ULL,
    0x8000400801980ULL,    0x2000810040200ULL,    0x8010100228810400ULL,0x2000009044210200ULL,
    0x4080008040102101ULL, 0x40002080411d01ULL,   0x2005524060000901ULL,0x502001008400422ULL,
    0x489a000810200402ULL, 0x1004400080a13ULL,    0x4000011008020084ULL,0x26002114058042ULL,
};

constexpr Bitboard BISHOP_MAGIC_NUMS[64] = {
    0x89a1121896040240ULL, 0x2004844802002010ULL, 0x2068080051921000ULL, 0x62880a0220200808ULL,
    0x4042004000000ULL,    0x100822020200011ULL,  0xc00444222012000aULL, 0x28808801216001ULL,
    0x400492088408100ULL,  0x201c401040c0084ULL,  0x840800910a0010ULL,   0x82080240060ULL,
    0x2000840504006000ULL, 0x30010c4108405004ULL, 0x1008005410080802ULL, 0x8144042209100900ULL,
    0x208081020014400ULL,  0x4800201208ca00ULL,   0xf18140408012008ULL,  0x1004002802102001ULL,
    0x841000820080811ULL,  0x40200200a42008ULL,   0x800054042000ULL,     0x88010400410c9000ULL,
    0x520040470104290ULL,  0x1004040051500081ULL, 0x2002081833080021ULL, 0x400c00c010142ULL,
    0x941408200c002000ULL, 0x658810000806011ULL,  0x188071040440a00ULL,  0x4800404002011c00ULL,
    0x104442040404200ULL,  0x511080202091021ULL,  0x4022401120400ULL,    0x80c0040400080120ULL,
    0x8040010040820802ULL, 0x480810700020090ULL,  0x102008e00040242ULL,  0x809005202050100ULL,
    0x8002024220104080ULL, 0x431008804142000ULL,  0x19001802081400ULL,   0x200014208040080ULL,
    0x3308082008200100ULL, 0x41010500040c020ULL,  0x4012020c04210308ULL, 0x208220a202004080ULL,
    0x111040120082000ULL,  0x6803040141280a00ULL, 0x2101004202410000ULL, 0x8200000041108022ULL,
    0x21082088000ULL,      0x2410204010040ULL,    0x40100400809000ULL,   0x822088220820214ULL,
    0x40808090012004ULL,   0x910224040218c9ULL,   0x402814422015008ULL,  0x90014004842410ULL,
    0x1000042304105ULL,    0x10008830412a00ULL,   0x2520081090008908ULL, 0x40102000a0a60140ULL,
};

// Compute the relevant occupancy mask for a sliding piece on `sq`.
// (All bits that could affect the attack set, EXCLUDING edge squares of each ray.)
Bitboard rook_mask(int sq) {
    Bitboard m = 0;
    const int r = sq >> 3, f = sq & 7;
    for (int rr = r + 1; rr <= 6; ++rr) m |= sq_bb(rr * 8 + f);
    for (int rr = r - 1; rr >= 1; --rr) m |= sq_bb(rr * 8 + f);
    for (int ff = f + 1; ff <= 6; ++ff) m |= sq_bb(r  * 8 + ff);
    for (int ff = f - 1; ff >= 1; --ff) m |= sq_bb(r  * 8 + ff);
    return m;
}

Bitboard bishop_mask(int sq) {
    Bitboard m = 0;
    const int r = sq >> 3, f = sq & 7;
    for (int rr = r + 1, ff = f + 1; rr <= 6 && ff <= 6; ++rr, ++ff) m |= sq_bb(rr * 8 + ff);
    for (int rr = r + 1, ff = f - 1; rr <= 6 && ff >= 1; ++rr, --ff) m |= sq_bb(rr * 8 + ff);
    for (int rr = r - 1, ff = f + 1; rr >= 1 && ff <= 6; --rr, ++ff) m |= sq_bb(rr * 8 + ff);
    for (int rr = r - 1, ff = f - 1; rr >= 1 && ff >= 1; --rr, --ff) m |= sq_bb(rr * 8 + ff);
    return m;
}

// Compute attacks given an occupancy by ray-casting (SLOW — used only at init).
Bitboard rook_attacks_slow(int sq, Bitboard occ) {
    Bitboard a = 0;
    const int r = sq >> 3, f = sq & 7;
    for (int rr = r + 1; rr <= 7; ++rr) { a |= sq_bb(rr * 8 + f); if (occ & sq_bb(rr * 8 + f)) break; }
    for (int rr = r - 1; rr >= 0; --rr) { a |= sq_bb(rr * 8 + f); if (occ & sq_bb(rr * 8 + f)) break; }
    for (int ff = f + 1; ff <= 7; ++ff) { a |= sq_bb(r  * 8 + ff); if (occ & sq_bb(r  * 8 + ff)) break; }
    for (int ff = f - 1; ff >= 0; --ff) { a |= sq_bb(r  * 8 + ff); if (occ & sq_bb(r  * 8 + ff)) break; }
    return a;
}

Bitboard bishop_attacks_slow(int sq, Bitboard occ) {
    Bitboard a = 0;
    const int r = sq >> 3, f = sq & 7;
    for (int rr = r + 1, ff = f + 1; rr <= 7 && ff <= 7; ++rr, ++ff) { a |= sq_bb(rr * 8 + ff); if (occ & sq_bb(rr * 8 + ff)) break; }
    for (int rr = r + 1, ff = f - 1; rr <= 7 && ff >= 0; ++rr, --ff) { a |= sq_bb(rr * 8 + ff); if (occ & sq_bb(rr * 8 + ff)) break; }
    for (int rr = r - 1, ff = f + 1; rr >= 0 && ff <= 7; --rr, ++ff) { a |= sq_bb(rr * 8 + ff); if (occ & sq_bb(rr * 8 + ff)) break; }
    for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; --rr, --ff) { a |= sq_bb(rr * 8 + ff); if (occ & sq_bb(rr * 8 + ff)) break; }
    return a;
}

// Iterate every subset of `mask` by Carry-Rippler trick.
// Used at init to populate magic attack tables.
void init_magic_for_square(int sq, bool is_bishop) {
    Magic& m = is_bishop ? BISHOP_MAGICS[sq] : ROOK_MAGICS[sq];
    m.mask  = is_bishop ? bishop_mask(sq) : rook_mask(sq);
    m.magic = is_bishop ? BISHOP_MAGIC_NUMS[sq] : ROOK_MAGIC_NUMS[sq];
    m.shift = 64 - popcount(m.mask);

    Bitboard occ = 0;
    do {
        const Bitboard attacks = is_bishop
            ? bishop_attacks_slow(sq, occ)
            : rook_attacks_slow(sq, occ);
#if ANDYBOT_AVX512
        const std::size_t idx = _pext_u64(occ, m.mask);
#else
        const std::size_t idx = (occ * m.magic) >> m.shift;
#endif
        m.attacks[idx] = attacks;
        occ = (occ - m.mask) & m.mask;  // next subset of mask
    } while (occ != 0);
}

void init_magics() {
    // Allocate slots in the global tables sequentially per square.
    Bitboard* b_ptr = BISHOP_TABLE;
    Bitboard* r_ptr = ROOK_TABLE;
    for (int sq = 0; sq < 64; ++sq) {
        BISHOP_MAGICS[sq].attacks = b_ptr;
        ROOK_MAGICS[sq].attacks   = r_ptr;
        const int b_size = 1 << popcount(bishop_mask(sq));
        const int r_size = 1 << popcount(rook_mask(sq));
        // Zero the slice (Carry-Rippler may not visit every index).
        for (int i = 0; i < b_size; ++i) b_ptr[i] = 0;
        for (int i = 0; i < r_size; ++i) r_ptr[i] = 0;
        init_magic_for_square(sq, true);
        init_magic_for_square(sq, false);
        b_ptr += b_size;
        r_ptr += r_size;
    }
}

// The AVX-512 build indexes with BMI2 pext (fast on every AVX-512 CPU); the AVX2 build keeps the magic multiply,
// because pext is microcoded and slow on older AMD processors.
inline Bitboard bishop_attacks(int sq, Bitboard occ) {
    const Magic& m = BISHOP_MAGICS[sq];
#if ANDYBOT_AVX512
    return m.attacks[_pext_u64(occ, m.mask)];
#else
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
#endif
}

inline Bitboard rook_attacks(int sq, Bitboard occ) {
    const Magic& m = ROOK_MAGICS[sq];
#if ANDYBOT_AVX512
    return m.attacks[_pext_u64(occ, m.mask)];
#else
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
#endif
}

inline Bitboard queen_attacks(int sq, Bitboard occ) {
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
}

void init_all() {
    init_knight_attacks();
    init_king_attacks();
    init_pawn_attacks();
    init_magics();
}

}  // namespace bb

#include "params.inc"
#include "network.inc"
#include "board.inc"
#include "movegen.inc"
#include "engine_support.inc"
class Engine {
public:
    #include "engine_api.inc"
private:
    #include "network_state.inc"
    #include "tt.inc"
    #include "network_eval.inc"
    #include "history.inc"
    #include "eval.inc"
    #include "search.inc"
    #include "engine_state.inc"
};

} // namespace

#include "uci.inc"
