// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

struct SearchConfig {
    int move_overhead_ms = ANDYBOT_DEFAULT_MOVE_OVERHEAD_MS;
};

struct PositionState {
    std::string fen = "startpos";
    std::vector<std::string> moves;
};



struct SearchResult {
    std::string bestmove = "0000";
    std::string ponder;  // expected reply, "" if unknown
    std::string pv;      // principal variation in UCI notation, starts with bestmove
    int seldepth = 0;
    int hashfull = 0;    // per mille
    int score_cp = 0;
    std::uint64_t nodes = 0;
    int depth = 0;
};

// weights are stored as IEEE-754 binary16 (uint16_t), biases as fp32.
// At each use site we load uint16_t and convert to fp32 via cvtph_ps.
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    [[nodiscard]] T* allocate(std::size_t count) {
        return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t(Alignment)));
    }
    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete(pointer, std::align_val_t(Alignment));
    }
    template <typename U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
};
template <typename T, typename U, std::size_t Alignment>
constexpr bool operator==(const AlignedAllocator<T, Alignment>&,
                          const AlignedAllocator<U, Alignment>&) noexcept { return true; }
template <typename T, typename U, std::size_t Alignment>
constexpr bool operator!=(const AlignedAllocator<T, Alignment>&,
                          const AlignedAllocator<U, Alignment>&) noexcept { return false; }

template <typename T> using Q8Vector = std::vector<T, AlignedAllocator<T, 64>>;
template <typename T> using EvalVector = std::vector<T>;

// One output head: everything after the shared accumulator. The network has eight, selected by piece count.
struct OutputHead {
    Q8Vector<std::int8_t> l1_weights;            // [256 groups][32][4]
    std::array<float, L1_OUT> l1_scale{};
    std::vector<float> l1_bias;                          // [L1_OUT], first 32 used
    EvalVector<float> l2_weights_t;                     // [64][32]
    std::array<float, 32> l2_bias{};
    std::array<float, 128> out_weights{};
    float out_bias = 0.0f;
};

struct NetworkWeights {
    std::array<OutputHead, 8> output_heads;
    std::vector<std::uint16_t> fc_us;        // [N_FEATURES][EMB_DIM] fp16, as stored in the file
    // Integer deployment copy of the HalfKAv2_hm feature table. Each output channel has its own
    // symmetric scale; accumulators store sums of the signed bytes in int16.
    Q8Vector<std::int8_t>       fc_us_q8;
    Q8Vector<std::int8_t>       fc_them_q8;
    std::array<float, EMB_DIM>  fc_us_q8_scale{};
    std::array<float, EMB_DIM>  fc_them_q8_scale{};
    std::array<std::int16_t, EMB_DIM> fc_us_clip_max{};
    std::array<std::int16_t, EMB_DIM> fc_them_clip_max{};
    // When every channel shares one clip value and one multiplier (all from-scratch nets do), the pairwise
    // product fits int16 and can be computed 32 lanes at a time.
    bool pair_uniform = false;
    std::int16_t pair_clip = 0;
    std::int16_t pair_mult = 0;
    // Q15 multipliers map a product of two clipped integer accumulators
    // directly into the existing [0,127] pairwise activation contract.
    std::array<std::int32_t, PAIRWISE_HALF> fc_us_pair_q15{};
    std::array<std::int32_t, PAIRWISE_HALF> fc_them_pair_q15{};
};

inline const Q8Vector<std::int8_t>& us_embedding_q8(const NetworkWeights& w) {
    return w.fc_us_q8;
}

// fp16 -> fp32 scalar conversion. Used only for tail elements / fallback.
// Uses a 256-bit-wide single-lane intrinsic.
inline float fp16_to_fp32(std::uint16_t h) {
    // Pack a single half into the low lane of __m128i, convert one float out.
    const __m128i v = _mm_cvtsi32_si128(static_cast<int>(h));
    const __m128 f  = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}


// Exact int16 pairwise is possible when all channels share one clip and one multiplier, the clipped product
// fits int16 and (p * mult + 2^14) >> 15 is what _mm512_mulhrs_epi16 computes.
inline void set_pair_uniform(NetworkWeights& out) {
    const int c = out.fc_us_clip_max[0];
    const std::int32_t m = out.fc_us_pair_q15[0];
    bool ok = c > 0 && c * c <= 32767 && m > 0 && m <= 32767;
    for (std::size_t j = 0; ok && j < EMB_DIM; ++j) ok = out.fc_us_clip_max[j] == c;
    for (std::size_t j = 0; ok && j < PAIRWISE_HALF; ++j) ok = out.fc_us_pair_q15[j] == m;
    out.pair_uniform = ok;
    out.pair_clip = static_cast<std::int16_t>(c);
    out.pair_mult = static_cast<std::int16_t>(m);
}

// The file stores the trained per-channel scales of the feature table. They are not derivable from the fp16
// values: recomputing max(abs(weight)) / 127 here would change the network.
inline void quantize_feature_table(
    NetworkWeights& out, const std::vector<float>& saved_scales) {
    if (saved_scales.size() != EMB_DIM) return;
    for (std::size_t j = 0; j < EMB_DIM; ++j)
        out.fc_us_q8_scale[j] = std::max(saved_scales[j], 1.0e-12f);
    constexpr float q15 = 32768.0f;
    out.fc_us_q8.resize(out.fc_us.size());
    for (std::size_t i = 0; i < N_FEATURES; ++i) {
        const std::size_t base = i * EMB_DIM;
        for (std::size_t j = 0; j < EMB_DIM; ++j) {
            const int q = static_cast<int>(std::nearbyint(
                fp16_to_fp32(out.fc_us[base + j]) / out.fc_us_q8_scale[j]));
            out.fc_us_q8[base + j] = static_cast<std::int8_t>(
                std::clamp(q, -127, 127));
        }
    }
    for (std::size_t j = 0; j < EMB_DIM; ++j)
        out.fc_us_clip_max[j] = static_cast<std::int16_t>(std::clamp(
            static_cast<int>(std::nearbyint(1.0f / out.fc_us_q8_scale[j])),
            0, 32767));
    for (std::size_t j = 0; j < PAIRWISE_HALF; ++j)
        out.fc_us_pair_q15[j] = static_cast<std::int32_t>(std::nearbyint(
            out.fc_us_q8_scale[j] * out.fc_us_q8_scale[j + PAIRWISE_HALF]
            * 127.0f * q15));
    out.fc_them_q8 = out.fc_us_q8;
    out.fc_them_q8_scale = out.fc_us_q8_scale;
    out.fc_them_clip_max = out.fc_us_clip_max;
    out.fc_them_pair_q15 = out.fc_us_pair_q15;
    set_pair_uniform(out);
}


std::vector<std::string> split_ws(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) {
        out.push_back(tok);
    }
    return out;
}

std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

static inline char lower_piece_ascii(char p) noexcept {
    return (p >= 'A' && p <= 'Z') ? static_cast<char>(p | 0x20) : p;
}

// Everything the hot paths derive from a mailbox piece char, indexed by the raw byte (no branches, no bounds check).
struct PieceInfo {
    std::int8_t code = -1, color = -1, kind = -1;   // code = color * 6 + kind
    std::uint8_t minor_major = 0;
    char upper = 0, lower = 0;
    std::int16_t pval = 0;                          // ordering value: P..Q 100/320/330/500/900, K and empty 0
    std::int16_t see = 0;                           // SEE value: as pval, K 10000
};
inline const std::array<PieceInfo, 256> PIECE_INFO = [] {
    std::array<PieceInfo, 256> t{};
    for (int c = 0; c < 256; ++c) {
        t[c].upper = static_cast<char>(c >= 'a' && c <= 'z' ? c - 32 : c);
        t[c].lower = static_cast<char>(c >= 'A' && c <= 'Z' ? c | 0x20 : c);
    }
    const char* letters = "pnbrqk";
    const int values[6] = {100, 320, 330, 500, 900, 0};
    const int see_values[6] = {100, 320, 330, 500, 900, 10000};
    for (int k = 0; k < 6; ++k) {
        for (int color = 0; color < 2; ++color) {
            const unsigned char ch = static_cast<unsigned char>(color == 0 ? letters[k] - 32 : letters[k]);
            t[ch].code = static_cast<std::int8_t>(color * 6 + k);
            t[ch].color = static_cast<std::int8_t>(color);
            t[ch].kind = static_cast<std::int8_t>(k);
            t[ch].minor_major = (k >= 1 && k <= 4) ? 1 : 0;
            t[ch].pval = static_cast<std::int16_t>(values[k]);
            t[ch].see = static_cast<std::int16_t>(see_values[k]);
        }
    }
    return t;
}();
inline const PieceInfo& piece_info(char c) { return PIECE_INFO[static_cast<unsigned char>(c)]; }

// White pieces are upper-case letters. The ASCII test equals std::isupper in the "C" locale
// (the engine never changes locale) but skips the CRT's locale checks on every call.
#define ANDYBOT_PIECE_IS_UPPER(c) ((c) >= 'A' && (c) <= 'Z')

// HalfKAv2_hm feature extraction.
//
// Matches SF nnue-pytorch DLL's HalfKAv2_hm exactly (NUM_PT=12, MIRROR FOLDED).
// Encoding (per SF tmp_nnue_ref/data_loader__cpp__training_data_loader.cpp):
//   p_idx     = piece_type * 2 + (piece_color != perspective_color)  // 0..11
//   o_sq      = orient_flip(perspective_color, sq, ksq)
//   o_ksq     = orient_flip(perspective_color, ksq, ksq)
//   feature   = o_sq + p_idx * 64 + KingBuckets[o_ksq] * (64*12)
// Where orient_flip:
//   bool h = (ksq.file() < e)
//   if perspective is Black:  sq ^= 56   (flip rank)
//   if h:                     sq ^=  7   (flip file)
//
// Piece type codes: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5.
//
// KingBuckets[64]: half the squares (files a-d) are -1; valid king squares are
// only e-h files after the orient flip pushes king to the e-h side.
constexpr int HALFKA_NUM_PT = 12;
constexpr int HALFKA_NUM_PLANES = 64 * HALFKA_NUM_PT;   // 768

// The bucket table is the HalfKAv2_hm layout of Stockfish / nnue-pytorch (GPL-3).
constexpr int HALFKA_KING_BUCKETS[64] = {
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1,  3,  2,  1,  0,
};

static inline int orient_flip_sq(int color /* 0=W, 1=B */, int sq, int ksq) {
    const bool h = (ksq & 7) < 4;       // ksq file < e
    int s = sq;
    if (color == 1) s ^= 56;            // flip rank for Black perspective
    if (h)          s ^=  7;            // flip file when king on a-d side
    return s;
}

// piece_type_code: P=0, N=1, B=2, R=3, Q=4, K=5. Returns -1 for unknown.
static inline int halfka_piece_type(char ch) {
    return piece_info(ch).kind;
    switch (lower_piece_ascii(ch)) {
        case 'p': return 0;
        case 'n': return 1;
        case 'b': return 2;
        case 'r': return 3;
        case 'q': return 4;
        case 'k': return 5;
        default:  return -1;
    }
}

// File layout ("AB21"): header, feature table (fp16) and its channel scales, then eight output heads, each
// 1024 -> 32 (int8 after loading), 64 -> 32 and 128 -> 1.
bool load_model_stream(std::istream& f, NetworkWeights& out) {
    char magic[4]{};
    f.read(magic, 4);
    if (!f || std::string(magic, 4) != "AB21") return false;

    auto read_u32 = [&f]() -> std::uint32_t {
        std::uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    const std::uint32_t version = read_u32();
    const std::uint32_t n_features = read_u32();
    const std::uint32_t emb_dim = read_u32();
    const std::uint32_t main1_in = read_u32();
    const std::uint32_t main1_out = read_u32();
    const std::uint32_t main2_in = read_u32();
    const std::uint32_t main2_out = read_u32();
    const std::uint32_t reserved_a = read_u32();
    const std::uint32_t reserved_b = read_u32();
    const std::uint32_t out_in = read_u32();
    const std::uint32_t out_out = read_u32();
    const std::uint32_t n_buckets = read_u32();
    if (!f || version != 1u || n_features != N_FEATURES || emb_dim != EMB_DIM || main1_in != EMB_DIM ||
        main1_out != 32u || main2_in != 64u || main2_out != 32u || reserved_a != 0u || reserved_b != 0u ||
        out_in != 128u || out_out != 1u || n_buckets != 8u) return false;

    auto read_fp16 = [&f](std::vector<std::uint16_t>& values, std::size_t count) {
        values.resize(count);
        f.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
        return static_cast<bool>(f);
    };
    auto read_f32 = [&f](std::vector<float>& values, std::size_t count) {
        values.resize(count);
        f.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
        return static_cast<bool>(f);
    };
    std::vector<float> shared_scales;
    if (!read_fp16(out.fc_us, static_cast<std::size_t>(N_FEATURES) * EMB_DIM) || !read_f32(shared_scales, EMB_DIM))
        return false;
    quantize_feature_table(out, shared_scales);

    // Per head: int8 first layer per output (rows grouped by four inputs), fp32 tail from fp16.
    for (std::size_t b = 0; b < 8; ++b) {
        std::vector<std::uint16_t> m1w, m2w, ow;
        std::vector<float> m1b, m2b, ob;
        if (!read_fp16(m1w, 32u * EMB_DIM) || !read_f32(m1b, 32u) || !read_fp16(m2w, 32u * 64u) || !read_f32(m2b, 32u) ||
            !read_fp16(ow, 128u) || !read_f32(ob, 1u)) return false;
        OutputHead& head = out.output_heads[b];
        head.l1_weights.assign(1024u * 32u, 0);
        head.l1_scale.fill(1.0f);
        head.l1_bias.assign(L1_OUT, 0.0f);
        for (std::size_t o = 0; o < 32; ++o) {
            float max_abs = 0.0f;
            for (std::size_t j = 0; j < EMB_DIM; ++j)
                max_abs = std::max(max_abs, std::abs(fp16_to_fp32(m1w[o * EMB_DIM + j])));
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            head.l1_scale[o] = scale;
            head.l1_bias[o] = m1b[o];
            for (std::size_t j = 0; j < EMB_DIM; ++j) {
                const int q = static_cast<int>(std::nearbyint(fp16_to_fp32(m1w[o * EMB_DIM + j]) / scale));
                head.l1_weights[(j / 4) * 128 + o * 4 + (j % 4)] = static_cast<std::int8_t>(std::clamp(q, -127, 127));
            }
        }
        head.l2_weights_t.resize(64u * 32u);
        for (std::size_t i = 0; i < 64; ++i)
            for (std::size_t o = 0; o < 32; ++o)
                head.l2_weights_t[i * 32 + o] = fp16_to_fp32(m2w[o * 64 + i]);
        std::copy(m2b.begin(), m2b.end(), head.l2_bias.begin());
        for (std::size_t i = 0; i < 128; ++i)
            head.out_weights[i] = fp16_to_fp32(ow[i]);
        head.out_bias = ob[0];
    }
    out.fc_them_q8.clear();
    out.fc_them_q8.shrink_to_fit();
    return true;
}

bool load_model_file(const std::filesystem::path& path, NetworkWeights& out) {
    std::ifstream f(path, std::ios::binary);
    return f && load_model_stream(f, out);
}

class ReadOnlyMemoryBuffer final : public std::streambuf {
public:
    ReadOnlyMemoryBuffer(const void* data, std::size_t size) {
        auto* begin = const_cast<char*>(static_cast<const char*>(data));
        setg(begin, begin, begin + size);
    }
};

// The network linked into the executable (see main.cpp).
bool load_embedded_model(NetworkWeights& out) {
    ReadOnlyMemoryBuffer buffer(andybot_net_begin, static_cast<std::size_t>(andybot_net_end - andybot_net_begin));
    std::istream stream(&buffer);
    return load_model_stream(stream, out);
}

// Grouped U8/S8 projection. The mathematical layer remains 1024->48.
// Layout is [input_group_4][output][four weights], allowing output-parallel
// accumulation and skipping groups whose four quantized activations are zero.
#include "network_projection.inc"
