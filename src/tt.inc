// AndyBot, a UCI chess engine. Copyright (C) 2026 Hyeonung Cho.
//
// AndyBot is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
// later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
// (the LICENSE file) for details. Credits to the engines AndyBot builds on are in README.md.

    // Lockless 4-way bucketed TT. Each bucket is exactly one cache line
    // (64 B). Slots use the standard XOR trick: store key XOR data so that
    // any torn write surfaces as a key mismatch on probe (cache miss, never
    // a wrong score).
    static constexpr int TT_BUCKET_N = 4;
    using TTWord = std::atomic<std::uint64_t>;
    static std::uint64_t tt_word_load(const TTWord& word) noexcept {
        return word.load(std::memory_order_relaxed);
    }
    static void tt_word_store(TTWord& word, std::uint64_t value) noexcept {
        word.store(value, std::memory_order_relaxed);
    }
    struct alignas(64) TTBucket {
        TTWord key_xor_data[TT_BUCKET_N];
        TTWord data[TT_BUCKET_N];
        TTBucket() noexcept {
            for (int i = 0; i < TT_BUCKET_N; ++i) {
                tt_word_store(key_xor_data[i], 0);
                tt_word_store(data[i], 0);
            }
        }
    };

    static constexpr int TT_NO_STATIC_EVAL = 32767;
    // Search generation in the top 4 bits of the packed word (unused by the
    // TT_STATIC_EVAL layout, which ends at bit 59).
    static constexpr int TT_GEN_SHIFT = 60;
    static constexpr int TT_GEN_MASK = 0xF;

    static std::uint64_t tt_pack(int depth, int score, int best_key,
                                 int static_eval, std::uint8_t bound) {
        const int s_clamped = std::clamp(score, -32000, 32000);
        const int e_clamped = static_eval == TT_NO_STATIC_EVAL
            ? TT_NO_STATIC_EVAL : std::clamp(static_eval, -32000, 32000);
        const std::uint64_t d = static_cast<std::uint16_t>(
            static_cast<std::int16_t>(std::clamp(depth, -512, 511))) & 0x3FFULL;
        const std::uint64_t s = static_cast<std::uint16_t>(static_cast<std::int16_t>(s_clamped));
        const std::uint64_t k = static_cast<std::uint16_t>(best_key & 0xFFFF);
        const std::uint64_t e = static_cast<std::uint16_t>(static_cast<std::int16_t>(e_clamped));
        return d | (s << 10) | (k << 26) | (e << 42)
             | ((static_cast<std::uint64_t>(bound) & 0x3ULL) << 58);
    }

    static int tt_packed_depth(std::uint64_t packed) noexcept {
        int depth = static_cast<int>(packed & 0x3FFULL);
        return (depth & 0x200) ? depth - 0x400 : depth;
    }

    static void tt_unpack(std::uint64_t packed, TTEntry& out) {
        out.depth = tt_packed_depth(packed);
        out.score = static_cast<std::int16_t>((packed >> 10) & 0xFFFF);
        out.best_key = static_cast<int>(static_cast<std::uint16_t>((packed >> 26) & 0xFFFF));
        out.static_eval = static_cast<std::int16_t>((packed >> 42) & 0xFFFF);
        out.bound = static_cast<std::uint8_t>((packed >> 58) & 0x3);
    }

    void init_tt(int mb) {
        tt_mb_ = std::max(1, mb);
        const std::size_t bytes = static_cast<std::size_t>(tt_mb_) * 1024ULL * 1024ULL;
        std::size_t buckets = 1;
        while ((buckets * sizeof(TTBucket)) <= bytes && buckets < (1ULL << 28)) buckets <<= 1;
        if (buckets > 1) buckets >>= 1;
        if (buckets < 256) buckets = 256;
        tt_ = std::shared_ptr<TTBucket[]>(new TTBucket[buckets]());
        tt_bucket_count_ = buckets;
        tt_bucket_mask_ = buckets - 1;
    }

    void clear_tt() {
        if (!tt_) return;
        for (std::size_t b = 0; b < tt_bucket_count_; ++b) {
            TTBucket& bucket = tt_[b];
            for (int i = 0; i < TT_BUCKET_N; ++i) {
                tt_word_store(bucket.key_xor_data[i], 0);
                tt_word_store(bucket.data[i], 0);
            }
        }
    }

    inline void tt_prefetch(std::uint64_t h) const noexcept {
#if defined(__AVX2__)
        if (tt_) {
            _mm_prefetch(reinterpret_cast<const char*>(
                &tt_[static_cast<std::size_t>(h) & tt_bucket_mask_]), _MM_HINT_T0);
        }
#else
        (void)h;
#endif
    }


    void tt_store(std::uint64_t h, const TTEntry& entry, int ply) {
        if (!tt_) return;
        TTBucket& bucket = tt_[static_cast<std::size_t>(h) & tt_bucket_mask_];
        int replace = 0;
        int min_depth = INT_MAX;
        for (int i = 0; i < TT_BUCKET_N; ++i) {
            const std::uint64_t d   = tt_word_load(bucket.data[i]);
            const std::uint64_t kxd = tt_word_load(bucket.key_xor_data[i]);
            if (d == 0) { replace = i; break; }
            if ((kxd ^ d) == h) {
                const int existing_depth = tt_packed_depth(d);
                if (entry.depth >= existing_depth || entry.bound == TT_EXACT
                    ) {
                    replace = i;
                    break;
                }
                return;
            }
            const int slot_depth = tt_packed_depth(d);
            if (slot_depth < min_depth) {
                min_depth = slot_depth;
                replace = i;
            }
        }
        const int tt_score = andybot::search_score::to_tt(entry.score, ply, MATE, MAX_PLY);
        const std::uint64_t packed = tt_pack(entry.depth, tt_score, entry.best_key,
                                             entry.static_eval, entry.bound)
            ;
        tt_word_store(bucket.data[replace], packed);
        tt_word_store(bucket.key_xor_data[replace], h ^ packed);
    }
