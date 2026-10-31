// Implements: include/ninfer/ops/mtp_round.h
// Match: validated request-major K=1..5 MTP round transition.
#include "ops/launcher/mtp_round.h"

#include "core/device.h"
#include "ops/kernel/mtp_round.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

void mtp_prepare_next_round_launch(const Tensor& verify_ids, const Tensor& next_anchors,
                                   const Tensor& accepted, const Tensor& updated_frontiers,
                                   const Tensor& remaining_budgets, const Tensor& licensed_counts,
                                   const Tensor& rope_deltas, Tensor& alignment_ids,
                                   Tensor& next_extents, Tensor& ar_positions,
                                   Tensor& ar_rope_positions, Tensor& ar_valid_columns,
                                   std::int32_t max_context, cudaStream_t stream) {
    constexpr int kBlock = 32;
    // AMD port diagnostic: NINFER_MTP_AR_POS_SHIFT offsets the draft AR chain's
    // cache position so the hypothesis can be tested without a rebuild.
    static const int ar_shift = [] {
        const char* v = std::getenv("NINFER_MTP_AR_POS_SHIFT");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    const int k          = verify_ids.ne[0] - 1;
    const int batch      = verify_ids.ne[1];
    const int ar_step_stride =
        static_cast<int>(ar_positions.nb[1] / static_cast<std::int64_t>(sizeof(std::int32_t)));
    const dim3 grid(static_cast<unsigned int>((k + kBlock) / kBlock),
                    static_cast<unsigned int>(batch));
    mtp_prepare_next_round_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(verify_ids.data),
        static_cast<const std::int32_t*>(next_anchors.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<const std::int32_t*>(updated_frontiers.data),
        static_cast<const std::int32_t*>(remaining_budgets.data),
        static_cast<const std::int32_t*>(licensed_counts.data),
        static_cast<const std::int32_t*>(rope_deltas.data),
        static_cast<std::int32_t*>(alignment_ids.data),
        static_cast<std::int32_t*>(next_extents.data),
        static_cast<std::int32_t*>(ar_positions.data),
        static_cast<std::int32_t*>(ar_rope_positions.data),
        static_cast<std::int32_t*>(ar_valid_columns.data), k, ar_step_stride, max_context, ar_shift);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
