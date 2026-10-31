#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

// ============================================================================
// AMD/HIP port of ops/common/mbarrier.cuh
//
// AMD has no hardware mbarrier. Emulation layout for the shared u64 "barrier":
//   bits  [0..15]  expected arrivals per phase
//   bits  [16..63] monotonically increasing arrival counter
// Phase p completes when counter/arrivals > p, so waiters spin on that
// condition. Only intra-CTA producer/consumer sync is supported (cluster
// semantics are not available on RDNA).
// ============================================================================

namespace ninfer::ops {
namespace detail {

__device__ __forceinline__ void spin_until_phase(std::uint64_t* barrier, std::uint32_t phase) {
    const std::uint32_t arrivals =
        static_cast<std::uint32_t>(*barrier & 0xFFFFull);
    for (;;) {
        const std::uint32_t count =
            static_cast<std::uint32_t>((*barrier >> 16) & 0xFFFFFFFFull);
        if (arrivals == 0 || count / arrivals > phase) return;
        __threadfence_block();
    }
}

} // namespace detail

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    // zero the counter, keep the expected-arrivals field
    *reinterpret_cast<volatile std::uint64_t*>(barrier) =
        (static_cast<std::uint64_t>(arrivals) & 0xFFFFull);
    __threadfence_block();
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
    detail::spin_until_phase(barrier, phase);
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
    __threadfence_block();
    atomicAdd(reinterpret_cast<unsigned long long*>(barrier), 1ull << 16);
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
    // AMD has no TMA transaction tracking; treat as a plain arrive.
    (void)bytes;
    cta_mbarrier_arrive(barrier);
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
    // cluster-scoped fence not applicable on RDNA
}

} // namespace ninfer::ops
