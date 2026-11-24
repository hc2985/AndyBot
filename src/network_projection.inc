// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

// Exact integer projection: 32 useful outputs, no padded AB19 lanes.
void project_l1(const std::int8_t* weights, const std::uint8_t* x,
                    const float* scales, const float* bias, float* out) {
#if defined(__AVX512F__) || defined(_M_AVX512F)
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, b0 = a0, b1 = a0;
    for (std::size_t g = 0; g < 256; g += 2) {
        std::uint32_t p, q;
        std::memcpy(&p, x + g * 4, 4); std::memcpy(&q, x + g * 4 + 4, 4);
        const auto u = _mm512_set1_epi32(static_cast<int>(p));
        const auto v = _mm512_set1_epi32(static_cast<int>(q));
        const auto w = weights + g * 128;
        a0 = _mm512_dpbusd_epi32(a0, u, _mm512_loadu_si512(w));
        a1 = _mm512_dpbusd_epi32(a1, u, _mm512_loadu_si512(w + 64));
        b0 = _mm512_dpbusd_epi32(b0, v, _mm512_loadu_si512(w + 128));
        b1 = _mm512_dpbusd_epi32(b1, v, _mm512_loadu_si512(w + 192));
    }
    const auto finish = [&](int o, __m512i a) {
        auto v = _mm512_mul_ps(_mm512_cvtepi32_ps(a), _mm512_set1_ps(1.0f / 127.0f));
        v = _mm512_mul_ps(v, _mm512_loadu_ps(scales + o));
        v = _mm512_add_ps(v, _mm512_loadu_ps(bias + o));
        _mm512_storeu_ps(out + o, _mm512_max_ps(_mm512_setzero_ps(), v));
    };
    finish(0, _mm512_add_epi32(a0,b0)); finish(16, _mm512_add_epi32(a1,b1));
#else
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    const auto ones = _mm256_set1_epi16(1);
    for (std::size_t g = 0; g < 256; ++g) {
        std::uint32_t p; std::memcpy(&p, x + g * 4, 4);
        const auto u = _mm256_set1_epi32(static_cast<int>(p));
        const auto w = weights + g * 128;
        const auto dot = [&](int o) {
            return _mm256_madd_epi16(_mm256_maddubs_epi16(u,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + o))), ones);
        };
        a0 = _mm256_add_epi32(a0, dot(0)); a1 = _mm256_add_epi32(a1, dot(32));
        a2 = _mm256_add_epi32(a2, dot(64)); a3 = _mm256_add_epi32(a3, dot(96));
    }
    const auto finish = [&](int o, __m256i a) {
        auto v = _mm256_mul_ps(_mm256_cvtepi32_ps(a), _mm256_set1_ps(1.0f / 127.0f));
        v = _mm256_mul_ps(v, _mm256_loadu_ps(scales + o));
        v = _mm256_add_ps(v, _mm256_loadu_ps(bias + o));
        _mm256_storeu_ps(out + o, _mm256_max_ps(_mm256_setzero_ps(), v));
    };
    finish(0,a0); finish(8,a1); finish(16,a2); finish(24,a3);
#endif
}

#if !ANDYBOT_AVX512 && defined(__AVX2__) && !(defined(__AVX512F__) || defined(_M_AVX512F))
// NNZ_LUT8[mask] = positions of the set bits of an 8-bit mask (padded), for building the non-zero group list.
alignas(32) static const std::array<std::array<std::int32_t, 8>, 256> NNZ_LUT8 = [] {
    std::array<std::array<std::int32_t, 8>, 256> t{};
    for (int m = 0; m < 256; ++m) { int k = 0; for (int b = 0; b < 8; ++b) if (m & (1 << b)) t[m][k++] = b; }
    return t;
}();
// project_l1's AVX2 arithmetic restricted to the listed (non-zero) groups. Integer-exact: skipped groups add 0.
void project_l1_sparse_avx2(const std::int8_t* weights, const std::uint8_t* x, const std::uint32_t* idx,
                              int n, const float* scales, const float* bias, float* out) {
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    __m256i b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    const auto ones = _mm256_set1_epi16(1);
    const auto step = [&](std::uint32_t g, __m256i& o0, __m256i& o1, __m256i& o2, __m256i& o3) {
        std::uint32_t p;
        std::memcpy(&p, x + g * 4, 4);
        const auto u = _mm256_set1_epi32(static_cast<int>(p));
        const auto w = weights + g * 128;
        o0 = _mm256_add_epi32(o0, _mm256_madd_epi16(_mm256_maddubs_epi16(
            u, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w))), ones));
        o1 = _mm256_add_epi32(o1, _mm256_madd_epi16(_mm256_maddubs_epi16(
            u, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32))), ones));
        o2 = _mm256_add_epi32(o2, _mm256_madd_epi16(_mm256_maddubs_epi16(
            u, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64))), ones));
        o3 = _mm256_add_epi32(o3, _mm256_madd_epi16(_mm256_maddubs_epi16(
            u, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 96))), ones));
    };
    int i = 0;
    for (; i + 2 <= n; i += 2) { step(idx[i], a0, a1, a2, a3); step(idx[i + 1], b0, b1, b2, b3); }
    if (i < n) step(idx[i], a0, a1, a2, a3);
    const auto finish = [&](int o, __m256i a) {
        auto v = _mm256_mul_ps(_mm256_cvtepi32_ps(a), _mm256_set1_ps(1.0f / 127.0f));
        v = _mm256_mul_ps(v, _mm256_loadu_ps(scales + o));
        v = _mm256_add_ps(v, _mm256_loadu_ps(bias + o));
        _mm256_storeu_ps(out + o, _mm256_max_ps(_mm256_setzero_ps(), v));
    };
    finish(0, _mm256_add_epi32(a0, b0)); finish(8, _mm256_add_epi32(a1, b1));
    finish(16, _mm256_add_epi32(a2, b2)); finish(24, _mm256_add_epi32(a3, b3));
}
#endif

#if ANDYBOT_AVX512 && (defined(__AVX512F__) || defined(_M_AVX512F))
// Same projection, visiting only the groups of four inputs listed in idx (the non-zero ones). Integer-exact.
void project_l1_sparse(const std::int8_t* weights, const std::uint8_t* x, const std::uint32_t* idx,
                         int n, const float* scales, const float* bias, float* out) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
    const auto step = [&](std::uint32_t g, __m512i& lo, __m512i& hi) {
        std::uint32_t p;
        std::memcpy(&p, x + g * 4, 4);
        const __m512i u = _mm512_set1_epi32(static_cast<int>(p));
        const std::int8_t* w = weights + g * 128;
        lo = _mm512_dpbusd_epi32(lo, u, _mm512_loadu_si512(w));
        hi = _mm512_dpbusd_epi32(hi, u, _mm512_loadu_si512(w + 64));
    };
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        step(idx[i], a0, a1); step(idx[i + 1], b0, b1); step(idx[i + 2], c0, c1); step(idx[i + 3], d0, d1);
    }
    for (; i < n; ++i) step(idx[i], a0, a1);
    const auto finish = [&](int o, __m512i a) {
        auto v = _mm512_mul_ps(_mm512_cvtepi32_ps(a), _mm512_set1_ps(1.0f / 127.0f));
        v = _mm512_mul_ps(v, _mm512_loadu_ps(scales + o));
        v = _mm512_add_ps(v, _mm512_loadu_ps(bias + o));
        _mm512_storeu_ps(out + o, _mm512_max_ps(_mm512_setzero_ps(), v));
    };
    finish(0, _mm512_add_epi32(_mm512_add_epi32(a0, b0), _mm512_add_epi32(c0, d0)));
    finish(16, _mm512_add_epi32(_mm512_add_epi32(a1, b1), _mm512_add_epi32(c1, d1)));
}
#endif
