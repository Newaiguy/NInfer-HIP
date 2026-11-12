#pragma once

#include <cstdint>
#include <cstdlib>

// Resident KV window bookkeeping (Leg B / kvmem port, W2).
//
// When the Main KV address space is smaller than the logical context, the oldest pages are
// slid out (kv_store::slide_prefix_pages) instead of being kept resident. After a slide the
// address space numbers its slots in COMPACT coordinates: slot i holds absolute token
// (window_base + i). Positions (RoPE) stay absolute; only slot/mask indices are compact.
//
// Single-sequence engine: one value for the whole device context. Zero means no slide has
// happened and compact == absolute.
inline std::uint32_t g_ninfer_kv_window_base_tokens = 0;

// Always-resident window in tokens (NINFER_RESIDENT_TOKENS); 0 = disabled, i.e. compact == absolute.
inline std::uint32_t ninfer_resident_window_tokens() {
    static const std::uint32_t tokens = [] {
        const char* env = std::getenv("NINFER_RESIDENT_TOKENS");
        return env != nullptr && env[0] != '\0'
                   ? static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10))
                   : 0U;
    }();
    return tokens;
}
