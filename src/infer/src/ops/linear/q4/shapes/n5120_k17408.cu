#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {

Q4Launch select_q4_n5120_k17408(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_ksplit<5120, 17408, 4>;
    if (tokens <= 8) return launch_q4_ksplit<5120, 17408, 8>;
    if (tokens <= 16) return launch_q4_ksplit<5120, 17408, 16>;
    if (tokens <= 24) return launch_q4_ksplit<5120, 17408, 24>;
    if (tokens <= 32) return launch_q4_ksplit<5120, 17408, 32>;
    if (tokens <= 96) return launch_q4_mma_r64_c128;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
