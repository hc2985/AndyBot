// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

#define PROMO_UPPER(p) (piece_info(p).upper)
#define PROMO_LOWER(p) (piece_info(p).lower)
constexpr int WHITE = 0;
constexpr int BLACK = 1;
constexpr std::uint8_t CWK = 1 << 0;
constexpr std::uint8_t CWQ = 1 << 1;
constexpr std::uint8_t CBK = 1 << 2;
constexpr std::uint8_t CBQ = 1 << 3;
constexpr int INF = 100000000;
constexpr int MATE = 30000;
constexpr const char* STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

const std::array<std::array<std::uint64_t, 64>, 12> ZOBRIST_PIECE = []() {
    std::array<std::array<std::uint64_t, 64>, 12> t{};
    std::uint64_t seed = 0x8a5cd789635d2dffULL;
    for (int pi = 0; pi < 12; ++pi) {
        for (int sq = 0; sq < 64; ++sq) {
            seed = splitmix64(seed);
            t[pi][sq] = seed;
        }
    }
    return t;
}();

const std::array<std::uint64_t, 4> ZOBRIST_CASTLING = []() {
    std::array<std::uint64_t, 4> t{};
    std::uint64_t seed = 0x3b2f6a95c4017ef1ULL;
    for (int i = 0; i < 4; ++i) {
        seed = splitmix64(seed);
        t[i] = seed;
    }
    return t;
}();

const std::array<std::uint64_t, 16> ZOBRIST_CASTLING_STATE = []() {
    std::array<std::uint64_t, 16> table{};
    for (int rights = 0; rights < 16; ++rights) {
        std::uint64_t value = 0;
        for (int bit = 0; bit < 4; ++bit)
            if (rights & (1 << bit)) value ^= ZOBRIST_CASTLING[bit];
        table[rights] = value;
    }
    return table;
}();

const std::array<std::uint64_t, 64> ZOBRIST_EP = []() {
    std::array<std::uint64_t, 64> t{};
    std::uint64_t seed = 0x91fa13792a4b53d7ULL;
    for (int sq = 0; sq < 64; ++sq) {
        seed = splitmix64(seed);
        t[sq] = seed;
    }
    return t;
}();

const std::uint64_t ZOBRIST_STM = splitmix64(0x5f8e6a44d31c0b29ULL);

#if ANDYBOT_AVX512
// 8-byte aligned Move: one load/store per copy (the 7-byte form stalled store forwarding on every picker return).
struct alignas(8) Move {
#else
struct Move {
#endif
#if ANDYBOT_AVX512
    std::int8_t from = -1;
    std::int8_t to = -1;
    char promo = 0;
    bool cap = false;
    bool ep = false;
    bool castle = false;
    bool dbl = false;
    std::uint8_t pad_ = 0;

    constexpr Move() noexcept = default;
    constexpr Move(int f, int t, char p, bool c, bool e, bool cs, bool d) noexcept
        : from(static_cast<std::int8_t>(f)), to(static_cast<std::int8_t>(t)),
          promo(p), cap(c), ep(e), castle(cs), dbl(d) {}
#else
    int from = -1;
    int to = -1;
    char promo = 0;
    bool cap = false;
    bool ep = false;
    bool castle = false;
    bool dbl = false;
#endif
};

// Stack-allocated, fixed-capacity move list. Chess positions max out at ~218
// pseudo-legal moves; 256 is the conservative ceiling used throughout the
// engine ecosystem. Exposes a minimal vector-like API so call sites that
// previously held std::vector<Move> can swap in MoveList with no other change.
// Replaces a per-ply heap allocation (which hits the allocator several times
// at every recursive search call).
struct MoveList {
    // The proven maximum number of legal chess moves is 218. Pseudo-legal
    // generation can include illegal king/pin moves, so retain six spare slots.
    static constexpr std::size_t CAPACITY = 224;
    static_assert(std::is_trivially_destructible_v<Move>,
                  "lazy MoveList storage requires trivial Move destruction");
    union Storage {
        Move data[CAPACITY];
        Storage() noexcept {}
        ~Storage() noexcept {}
    } storage_;
    std::size_t size_ = 0;

    inline Move* data_ptr() noexcept { return storage_.data; }
    inline const Move* data_ptr() const noexcept { return storage_.data; }

    inline std::size_t size() const noexcept { return size_; }
    inline bool empty() const noexcept { return size_ == 0; }
    inline void clear() noexcept { size_ = 0; }
    inline void resize(std::size_t n) noexcept { size_ = n; }
    // reserve() is a no-op; capacity is fixed at construction.
    inline void reserve(std::size_t) noexcept {}

    inline Move& operator[](std::size_t i) noexcept { return data_ptr()[i]; }
    inline const Move& operator[](std::size_t i) const noexcept { return data_ptr()[i]; }
    inline Move& front() noexcept { return data_ptr()[0]; }
    inline const Move& front() const noexcept { return data_ptr()[0]; }

    inline void push_back(const Move& m) noexcept {
        ::new (static_cast<void*>(data_ptr() + size_)) Move(m);
        ++size_;
    }
    inline void push_back(Move&& m) noexcept {
        ::new (static_cast<void*>(data_ptr() + size_)) Move(std::move(m));
        ++size_;
    }

    inline Move* begin() noexcept { return data_ptr(); }
    inline Move* end() noexcept { return data_ptr() + size_; }
    inline const Move* begin() const noexcept { return data_ptr(); }
    inline const Move* end() const noexcept { return data_ptr() + size_; }
};

struct Undo {
    Move m;
    char moved = '.';
    char captured = '.';
    // Piece-code cache for the incremental bitboard restore path. 255 means
    // empty/invalid; otherwise code = color * 6 + kind.
    std::uint8_t moved_code = 255;
    std::uint8_t captured_code = 255;
    std::uint8_t castling = 0;
    int ep = -1;
    int halfmove = 0;
    int fullmove = 1;
    int kw = -1;
    int kb = -1;
    std::uint64_t hash = 0ULL;
    // Bitboard snapshot taken at the start of makemove. unmakemove restores
    // these directly — much faster than recomputing incremental deltas in reverse.
};

struct Board {
    std::array<char, 64> sq{};
    int stm = WHITE;
    std::uint8_t castling = 0;
    int ep = -1;
    int halfmove = 0;
    int fullmove = 1;
    int king[2] = {-1, -1};
    std::uint64_t hash = 0ULL;

    // Bitboard view — kept in sync with sq[] by parse_fen + make/unmake.
    // pieces[color][kind]: kind = 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king.
    bb::Bitboard pieces[2][6] = {};
    bb::Bitboard color_occ[2] = {0, 0};
    bb::Bitboard occ = 0;
};

// Map mailbox piece char -> (color, kind). kind: 0=P,1=N,2=B,3=R,4=Q,5=K.
const std::array<std::int8_t, 128> PIECE_CODE = []() {
    std::array<std::int8_t, 128> table{};
    table.fill(-1);
    table['P'] = 0; table['N'] = 1; table['B'] = 2;
    table['R'] = 3; table['Q'] = 4; table['K'] = 5;
    table['p'] = 6; table['n'] = 7; table['b'] = 8;
    table['r'] = 9; table['q'] = 10; table['k'] = 11;
    return table;
}();

inline int board_piece_code(char c) {
    return piece_info(c).code;
    const unsigned int index = static_cast<unsigned char>(c);
    return index < PIECE_CODE.size() ? PIECE_CODE[index] : -1;
}

inline bool piece_to_bb_kind(char c, int& color_out, int& kind_out) {
    const PieceInfo& info = piece_info(c);
    color_out = info.color;
    kind_out = info.kind;
    return info.code >= 0;
}

// Rebuild bitboards from sq[] — call after parse_fen or any non-incremental change.
void sync_bitboards(Board& b) {
    for (int c = 0; c < 2; ++c) {
        for (int k = 0; k < 6; ++k) b.pieces[c][k] = 0;
        b.color_occ[c] = 0;
    }
    b.occ = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int color, kind;
        if (piece_to_bb_kind(b.sq[sq], color, kind)) {
            const bb::Bitboard mask = bb::sq_bb(sq);
            b.pieces[color][kind] |= mask;
            b.color_occ[color] |= mask;
            b.occ |= mask;
        }
    }
}

std::uint64_t hashb(const Board& b);
std::uint64_t compute_full_hash(const Board& b);

constexpr int MAX_PLY = 256;
constexpr std::uint8_t TT_EXACT = 0;
constexpr std::uint8_t TT_LOWER = 1;
constexpr std::uint8_t TT_UPPER = 2;

struct TTEntry {
    int depth = -1;
    int score = 0;
    int best_key = -1;
    int static_eval = 32767;
    std::uint8_t bound = TT_EXACT;
};

int file_of(int s) { return s & 7; }
int rank_of(int s) { return s >> 3; }
bool onb(int s) { return s >= 0 && s < 64; }

bool isw(char p) { return p >= 'A' && p <= 'Z'; }
bool isb(char p) { return p >= 'a' && p <= 'z'; }
int col(char p) { if (isw(p)) return WHITE; if (isb(p)) return BLACK; return -1; }

int pval(char p) {
    return piece_info(p).pval;
    switch (lower_piece_ascii(p)) {
        case 'p': return 100;
        case 'n': return 320;
        case 'b': return 330;
        case 'r': return 500;
        case 'q': return 900;
        default: return 0;
    }
}
int promo_code(char p) {
    switch (lower_piece_ascii(p)) {
        case 'q': return 1;
        case 'r': return 2;
        case 'b': return 3;
        case 'n': return 4;
        default: return 0;
    }
}

int move_key(const Move& m) {
    return (m.from & 63) | ((m.to & 63) << 6) | ((promo_code(m.promo) & 7) << 12);
}


// Maps a piece char to a 0..11 index for continuation-history indexing.
// Returns -1 for empty/unknown.
inline int piece_index(char p) {
    return board_piece_code(p);  // same 0..11 codes, -1 otherwise
    switch (p) {
        case 'P': return 0;  case 'N': return 1;  case 'B': return 2;
        case 'R': return 3;  case 'Q': return 4;  case 'K': return 5;
        case 'p': return 6;  case 'n': return 7;  case 'b': return 8;
        case 'r': return 9;  case 'q': return 10; case 'k': return 11;
        default:  return -1;
    }
}


std::string sq_to_uci(int s) {
    std::string o;
    o.push_back(static_cast<char>('a' + file_of(s)));
    o.push_back(static_cast<char>('1' + rank_of(s)));
    return o;
}

int uci_to_sq(const std::string& s) {
    if (s.size() != 2) return -1;
    if (s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8') return -1;
    return (s[1] - '1') * 8 + (s[0] - 'a');
}

std::string mv_to_uci(const Move& m) {
    if (!onb(m.from) || !onb(m.to)) return "0000";
    std::string s = sq_to_uci(m.from) + sq_to_uci(m.to);
    if (m.promo) s.push_back(PROMO_LOWER(m.promo));
    return s;
}

bool parse_fen(const std::string& fen_in, Board& b) {
    const std::string fen = (fen_in == "startpos") ? STARTPOS_FEN : fen_in;
    auto parts = split_ws(fen);
    if (parts.size() < 2) return false;
    b = Board{};
    b.sq.fill('.');

    std::vector<std::string> ranks;
    { std::stringstream ss(parts[0]); std::string r; while (std::getline(ss, r, '/')) ranks.push_back(r); }
    if (ranks.size() != 8) return false;

    for (int fr = 0; fr < 8; ++fr) {
        int f = 0;
        for (char ch : ranks[fr]) {
            if (std::isdigit(static_cast<unsigned char>(ch)) != 0) { f += (ch - '0'); continue; }
            if (f < 0 || f > 7) return false;
            int r = 7 - fr;
            int s = r * 8 + f;
            b.sq[s] = ch;
            if (ch == 'K') b.king[WHITE] = s;
            if (ch == 'k') b.king[BLACK] = s;
            ++f;
        }
        if (f != 8) return false;
    }
    if (b.king[WHITE] < 0 || b.king[BLACK] < 0) return false;

    b.stm = (parts[1] == "b") ? BLACK : WHITE;
    if (parts.size() > 2) {
        if (parts[2].find('K') != std::string::npos) b.castling |= CWK;
        if (parts[2].find('Q') != std::string::npos) b.castling |= CWQ;
        if (parts[2].find('k') != std::string::npos) b.castling |= CBK;
        if (parts[2].find('q') != std::string::npos) b.castling |= CBQ;
    }
    b.ep = (parts.size() > 3 && parts[3] != "-") ? uci_to_sq(parts[3]) : -1;
    b.halfmove = (parts.size() > 4) ? std::max(0, std::atoi(parts[4].c_str())) : 0;
    b.fullmove = (parts.size() > 5) ? std::max(1, std::atoi(parts[5].c_str())) : 1;
    b.hash = compute_full_hash(b);
    sync_bitboards(b);
    return true;
}

int zobrist_piece_index(char p) {
    return board_piece_code(p);
}

void hash_toggle_piece(std::uint64_t& h, char p, int sq) {
    if (!onb(sq) || p == '.') return;
    const int idx = zobrist_piece_index(p);
    if (idx >= 0) h ^= ZOBRIST_PIECE[idx][sq];
}

void hash_toggle_castling(std::uint64_t& h, std::uint8_t castling) {
    h ^= ZOBRIST_CASTLING_STATE[castling & 15];
}

void hash_toggle_ep(std::uint64_t& h, int ep_sq) {
    if (onb(ep_sq)) h ^= ZOBRIST_EP[ep_sq];
}

void hash_toggle_stm(std::uint64_t& h) {
    h ^= ZOBRIST_STM;
}

std::uint64_t compute_full_hash(const Board& b) {
    std::uint64_t h = 0ULL;
    for (int sq = 0; sq < 64; ++sq) {
        hash_toggle_piece(h, b.sq[sq], sq);
    }
    hash_toggle_castling(h, b.castling);
    hash_toggle_ep(h, b.ep);
    if (b.stm == BLACK) hash_toggle_stm(h);
    return h;
}

bool attacked(const Board& b, int sq, int by) {
    int f = file_of(sq), r = rank_of(sq);

    if (by == WHITE) {
        if (f > 0) { int s = sq - 9; if (onb(s) && b.sq[s] == 'P') return true; }
        if (f < 7) { int s = sq - 7; if (onb(s) && b.sq[s] == 'P') return true; }
    } else {
        if (f > 0) { int s = sq + 7; if (onb(s) && b.sq[s] == 'p') return true; }
        if (f < 7) { int s = sq + 9; if (onb(s) && b.sq[s] == 'p') return true; }
    }

    static const int ndf[8] = {1, 2, 2, 1, -1, -2, -2, -1};
    static const int ndr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    char n = (by == WHITE) ? 'N' : 'n';
    for (int i = 0; i < 8; ++i) {
        int nf = f + ndf[i], nr = r + ndr[i];
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        if (b.sq[nr * 8 + nf] == n) return true;
    }

    char k = (by == WHITE) ? 'K' : 'k';
    for (int df = -1; df <= 1; ++df) for (int dr = -1; dr <= 1; ++dr) {
        if (!df && !dr) continue;
        int nf = f + df, nr = r + dr;
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        if (b.sq[nr * 8 + nf] == k) return true;
    }

    auto ray = [&](int df, int dr, std::string tg) {
        int nf = f + df, nr = r + dr;
        while (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) {
            char p = b.sq[nr * 8 + nf];
            if (p != '.') return tg.find(p) != std::string::npos;
            nf += df; nr += dr;
        }
        return false;
    };

    if (by == WHITE) {
        if (ray(1,1,"BQ")||ray(1,-1,"BQ")||ray(-1,1,"BQ")||ray(-1,-1,"BQ")) return true;
        if (ray(1,0,"RQ")||ray(-1,0,"RQ")||ray(0,1,"RQ")||ray(0,-1,"RQ")) return true;
    } else {
        if (ray(1,1,"bq")||ray(1,-1,"bq")||ray(-1,1,"bq")||ray(-1,-1,"bq")) return true;
        if (ray(1,0,"rq")||ray(-1,0,"rq")||ray(0,1,"rq")||ray(0,-1,"rq")) return true;
    }

    return false;
}

// Forward declaration; the bitboard implementation is defined below alongside
// the bitboard movegen. Hot path: called 2-3x per search node (legality test,
// NMP gate, gives-check probe), so we delegate to the magic-bitboard attacker
// query instead of the slow raycast attacked() function.
bool is_square_attacked_bb(const Board& b, int sq, int by_color);
__forceinline
bool incheck(const Board& b, int c) {
    return is_square_attacked_bb(b, b.king[c], c ^ 1);
}

// SEE helpers: find the smallest attacker of `to` for side `side` on board
// `b`. Returns piece value (P=100/N=320/B=330/R=500/Q=900/K=10000) and
// writes source square to `out_from`. Returns 0 if no attacker.
inline int see_piece_value(char p) {
    return piece_info(p).see;
    switch (lower_piece_ascii(p)) {
        case 'p': return 100;
        case 'n': return 320;
        case 'b': return 330;
        case 'r': return 500;
        case 'q': return 900;
        case 'k': return 10000;
        default:  return 0;
    }
}

// Helpers for incremental bitboard updates. Each call flips one bit in the
// piece bitboard, color occupancy, and total occupancy.
inline void bb_remove(Board& b, int color, int kind, int sq) {
    const bb::Bitboard mask = bb::sq_bb(sq);
    b.pieces[color][kind] &= ~mask;
    b.color_occ[color]    &= ~mask;
    b.occ                 &= ~mask;
}
inline void bb_add(Board& b, int color, int kind, int sq) {
    const bb::Bitboard mask = bb::sq_bb(sq);
    b.pieces[color][kind] |= mask;
    b.color_occ[color]    |= mask;
    b.occ                 |= mask;
}

void makemove(Board& b, const Move& m, Undo& u) {
    u.m = m; u.castling = b.castling; u.ep = b.ep; u.halfmove = b.halfmove; u.fullmove = b.fullmove; u.kw = b.king[WHITE]; u.kb = b.king[BLACK]; u.hash = b.hash;
    u.moved = b.sq[m.from];
    u.captured = m.ep ? b.sq[(b.stm == WHITE) ? (m.to - 8) : (m.to + 8)] : b.sq[m.to];
    {
        int code_color = 0;
        int code_kind = 0;
        if (piece_to_bb_kind(u.moved, code_color, code_kind))
            u.moved_code = static_cast<std::uint8_t>(code_color * 6 + code_kind);
        else
            u.moved_code = 255;
        if (piece_to_bb_kind(u.captured, code_color, code_kind))
            u.captured_code = static_cast<std::uint8_t>(code_color * 6 + code_kind);
        else
            u.captured_code = 255;
    }

    int us = b.stm;
    std::uint64_t h = b.hash;
    hash_toggle_stm(h);
    hash_toggle_ep(h, b.ep);
    hash_toggle_castling(h, b.castling);

    hash_toggle_piece(h, u.moved, m.from);
    if (u.captured != '.') {
        const int cap_sq = m.ep ? ((us == WHITE) ? (m.to - 8) : (m.to + 8)) : m.to;
        hash_toggle_piece(h, u.captured, cap_sq);
    }

    b.ep = -1;
    b.sq[m.from] = '.';
    if (m.ep) b.sq[(us == WHITE) ? (m.to - 8) : (m.to + 8)] = '.';

    char put = u.moved;
    if (m.promo) put = (us == WHITE) ? PROMO_UPPER(m.promo) : PROMO_LOWER(m.promo);
    b.sq[m.to] = put;
    hash_toggle_piece(h, put, m.to);

    if (m.castle) {
        if (m.to == 6) {
            const char rook = b.sq[7];
            hash_toggle_piece(h, rook, 7);
            b.sq[5] = rook; b.sq[7] = '.';
            hash_toggle_piece(h, rook, 5);
        }
        else if (m.to == 2) {
            const char rook = b.sq[0];
            hash_toggle_piece(h, rook, 0);
            b.sq[3] = rook; b.sq[0] = '.';
            hash_toggle_piece(h, rook, 3);
        }
        else if (m.to == 62) {
            const char rook = b.sq[63];
            hash_toggle_piece(h, rook, 63);
            b.sq[61] = rook; b.sq[63] = '.';
            hash_toggle_piece(h, rook, 61);
        }
        else if (m.to == 58) {
            const char rook = b.sq[56];
            hash_toggle_piece(h, rook, 56);
            b.sq[59] = rook; b.sq[56] = '.';
            hash_toggle_piece(h, rook, 59);
        }
    }

    if (lower_piece_ascii(u.moved) == 'k') b.king[us] = m.to;
    if (m.dbl) b.ep = (us == WHITE) ? (m.from + 8) : (m.from - 8);

    if (u.moved == 'K') b.castling &= static_cast<std::uint8_t>(~(CWK | CWQ));
    if (u.moved == 'k') b.castling &= static_cast<std::uint8_t>(~(CBK | CBQ));
    if (u.moved == 'R') { if (m.from == 0) b.castling &= static_cast<std::uint8_t>(~CWQ); if (m.from == 7) b.castling &= static_cast<std::uint8_t>(~CWK); }
    if (u.moved == 'r') { if (m.from == 56) b.castling &= static_cast<std::uint8_t>(~CBQ); if (m.from == 63) b.castling &= static_cast<std::uint8_t>(~CBK); }

    if (!m.ep) {
        if (u.captured == 'R') { if (m.to == 0) b.castling &= static_cast<std::uint8_t>(~CWQ); if (m.to == 7) b.castling &= static_cast<std::uint8_t>(~CWK); }
        if (u.captured == 'r') { if (m.to == 56) b.castling &= static_cast<std::uint8_t>(~CBQ); if (m.to == 63) b.castling &= static_cast<std::uint8_t>(~CBK); }
    }

    b.halfmove = (lower_piece_ascii(u.moved) == 'p' || u.captured != '.') ? 0 : (b.halfmove + 1);
    if (us == BLACK) b.fullmove += 1;
    b.stm ^= 1;
    hash_toggle_castling(h, b.castling);
    hash_toggle_ep(h, b.ep);
    b.hash = h;

    // Incremental bitboard update — replaces brute-force sync_bitboards.
    // Reads the now-updated mailbox sq[] to determine final piece kinds (handles promotion).
    {
        const int mover_color = us;  // captured value: us was b.stm at start of move
        int mover_kind, _c;
        // u.moved tells us the original mover; the destination piece kind is whatever
        // is on m.to NOW (after promotion/no-promotion). For captures, the captured
        // piece had its square zeroed in sq[] but we still know its kind from u.captured.
        const bool have_mover = u.moved_code != 255;
        if (have_mover) {
            _c = u.moved_code / 6;
            mover_kind = u.moved_code % 6;
        }
        if (have_mover) {
            // Remove mover from FROM (always — even castling/promo/etc.).
            bb_remove(b, mover_color, mover_kind, m.from);

            // Remove captured piece (if any). For en passant, captured square != m.to.
            if (u.captured != '.') {
                int cap_color, cap_kind;
                const bool have_capture = u.captured_code != 255;
                if (have_capture) {
                    cap_color = u.captured_code / 6;
                    cap_kind = u.captured_code % 6;
                    int cap_sq = m.to;
                    if (m.ep) cap_sq = (mover_color == WHITE) ? (m.to - 8) : (m.to + 8);
                    bb_remove(b, cap_color, cap_kind, cap_sq);
                }
            }

            // Add destination piece (mover, or promoted piece) at TO.
            int dest_kind = mover_kind;
            if (m.promo) {
                const char puc = PROMO_UPPER(m.promo);
                switch (puc) {
                    case 'Q': dest_kind = 4; break;
                    case 'R': dest_kind = 3; break;
                    case 'B': dest_kind = 2; break;
                    case 'N': dest_kind = 1; break;
                    default:  break;
                }
            }
            bb_add(b, mover_color, dest_kind, m.to);

            // Castling: also relocate the rook. (King move was handled above.)
            if (m.castle) {
                int rfrom = -1, rto = -1;
                if      (m.to == 6)  { rfrom = 7;  rto = 5;  } // white O-O
                else if (m.to == 2)  { rfrom = 0;  rto = 3;  } // white O-O-O
                else if (m.to == 62) { rfrom = 63; rto = 61; } // black O-O
                else if (m.to == 58) { rfrom = 56; rto = 59; } // black O-O-O
                if (rfrom >= 0) {
                    bb_remove(b, mover_color, 3 /*rook*/, rfrom);
                    bb_add   (b, mover_color, 3 /*rook*/, rto);
                }
            }
        }
    }
}

void unmakemove(Board& b, const Undo& u) {
    const Move& m = u.m;
    b.stm ^= 1;
#define UNMAKE_STM b.stm
    b.castling = u.castling; b.ep = u.ep; b.halfmove = u.halfmove; b.fullmove = u.fullmove; b.king[WHITE] = u.kw; b.king[BLACK] = u.kb;

    if (m.castle) {
        if (m.to == 6) { b.sq[7] = b.sq[5]; b.sq[5] = '.'; }
        else if (m.to == 2) { b.sq[0] = b.sq[3]; b.sq[3] = '.'; }
        else if (m.to == 62) { b.sq[63] = b.sq[61]; b.sq[61] = '.'; }
        else if (m.to == 58) { b.sq[56] = b.sq[59]; b.sq[59] = '.'; }
    }

    b.sq[m.from] = u.moved;
    if (m.ep) {
        b.sq[m.to] = '.';
        b.sq[(UNMAKE_STM == WHITE) ? (m.to - 8) : (m.to + 8)] = u.captured;
    } else {
        b.sq[m.to] = u.captured;
    }
    b.hash = u.hash;

    // Restore bitboards from snapshot — O(1) instead of brute-force rebuild.
    const int mover_color = UNMAKE_STM;
    int ignored_color = 0;
    int mover_kind = 0;
    const bool have_mover = u.moved_code != 255;
    if (have_mover) {
        ignored_color = u.moved_code / 6;
        mover_kind = u.moved_code % 6;
        int destination_kind = mover_kind;
        if (m.promo) {
            switch (PROMO_UPPER(m.promo)) {
                case 'Q': destination_kind = 4; break;
                case 'R': destination_kind = 3; break;
                case 'B': destination_kind = 2; break;
                case 'N': destination_kind = 1; break;
                default: break;
            }
        }
        bb_remove(b, mover_color, destination_kind, m.to);
        bb_add(b, mover_color, mover_kind, m.from);
        if (u.captured != '.') {
            int captured_color = 0;
            int captured_kind = 0;
            const bool have_capture = u.captured_code != 255;
            if (have_capture) {
                captured_color = u.captured_code / 6;
                captured_kind = u.captured_code % 6;
                const int captured_square = m.ep
                    ? (mover_color == WHITE ? m.to - 8 : m.to + 8)
                    : m.to;
                bb_add(b, captured_color, captured_kind, captured_square);
            }
        }
        if (m.castle) {
            int rook_from = -1;
            int rook_to = -1;
            if      (m.to == 6)  { rook_from = 7;  rook_to = 5;  }
            else if (m.to == 2)  { rook_from = 0;  rook_to = 3;  }
            else if (m.to == 62) { rook_from = 63; rook_to = 61; }
            else if (m.to == 58) { rook_from = 56; rook_to = 59; }
            if (rook_from >= 0) {
                bb_remove(b, mover_color, 3, rook_to);
                bb_add(b, mover_color, 3, rook_from);
            }
        }
    }
}
#undef UNMAKE_STM

// Forward-decl: bitboard movegen defined below; the wrapper at the end of this
// file's mailbox version delegates to it so all search callers go through the
// bitboard path.
