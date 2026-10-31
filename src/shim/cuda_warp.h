#pragma once
// NInfer -> HIP port shim: warp intrinsics with 32-bit masks (NVIDIA style)
// AMD ignores shuffle masks (implicit convergence on wave32), so we forward
// a full 64-bit mask. Non-template mask param keeps these more specialized
// than HIP's 64-bit-mask templates.
#include <hip/hip_runtime.h>

// MSVC host compiles (engine .cpp via cl) don't get warpSize from HIP device headers
#ifndef warpSize
#define warpSize 32
#endif

// Warp intrinsic shims are device-only; pure host .cpp builds (MSVC) never need them
#if defined(__HIPCC__)

template <typename T>
__device__ __forceinline__ T __shfl_sync(unsigned, T var, int srcLane, int width = warpSize) {
    return ::__shfl_sync(0xffffffffull, var, srcLane, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned, T var, unsigned delta, int width = warpSize) {
    return ::__shfl_down_sync(0xffffffffull, var, delta, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_up_sync(unsigned, T var, unsigned delta, int width = warpSize) {
    return ::__shfl_up_sync(0xffffffffull, var, delta, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned, T var, int laneMask, int width = warpSize) {
    return ::__shfl_xor_sync(0xffffffffull, var, laneMask, width);
}
__device__ __forceinline__ unsigned __ballot_sync(unsigned, int pred) {
    return ::__ballot(pred);
}
__device__ __forceinline__ void __syncwarp(unsigned) {
    // AMD port: PTX bar.warp.sync has no real ROCm lowering here - the default
    // __nvvm_bar_warp_sync path compiles to nothing on amdgcn, which silently broke
    // every cross-lane shared-memory staging pattern (a lane writes a shared tile,
    // other lanes read it via ldmatrix-style loads): the attention softmax P tiles
    // were read before they were written and the whole attention output became
    // exactly zero. On wave32 a converged wave executes SIMT-synchronously, so a
    // wave-wide LDS drain fence reproduces the cross-lane ordering guarantee; the
    // memory clobber also stops the compiler from sinking the cross-lane LDS loads
    // above the fence (they alias the staged tile only through a shuffled address
    // the compiler cannot see).
    __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}
// Partial-mask warp max (CUDA semantics: max over lanes selected by mask, result
// broadcast to all participating lanes). AMD's __reduce_max_sync ignores the mask and
// reduces the full wave, which breaks when non-participating lanes diverged away —
// and its 64-bit-mask static_assert rejects CUDA-style 32-bit masks. Distinct name
// avoids clashing with HIP's __reduce_max_sync overload set. Assumes the mask selects
// contiguous lanes starting at 0 (true for ninfer sampling_device.cuh kMergeMask).
template <typename T>
__device__ __forceinline__ T ninfer_shim_reduce_max_sync(unsigned mask, T var) {
    const unsigned lane = static_cast<unsigned>(threadIdx.x) & 31u;
    for (unsigned d = 1u; d < 32u; d <<= 1u) {
        const unsigned src = lane + d;
        if (src < 32u && (mask & (1u << src)) != 0u) {
            const T v = ::__shfl_sync(0xffffffffull, var, static_cast<int>(src), 32);
            if (var < v) { var = v; }
        }
    }
    const int first = (mask != 0u) ? (__ffs(mask) - 1) : 0;
    return ::__shfl_sync(0xffffffffull, var, first, 32);
}
#endif
