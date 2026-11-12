#pragma once

#include <cstdint>

namespace ninfer::ops {

// Experimental slot -> absolute token position map consumed by the decode
// attention kernels (see CausalSmallTInvocation::key_position_map). Kept here,
// rather than in the op signature, so enabling recall does not force a rebuild
// of every translation unit that includes the attention op headers.
inline const std::int32_t* g_attention_key_position_map = nullptr;
inline std::int32_t g_attention_key_slots = 0;

}  // namespace ninfer::ops
