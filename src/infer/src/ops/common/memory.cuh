#pragma once

// ============================================================================
// AMD/HIP port of ops/common/memory.cuh (was: CUDA cp.async pipeline)
//
// RDNA has no hardware global->shared async copy. Port strategy:
//   cp_async / pipe_copy  -> plain vectorized loads+stores (compiler pipelines
//                            these well on RDNA4; global loads go through L2/L1)
//   cp_commit / cp_wait   -> no-ops; ordering across warps relies on the
//                            __syncthreads() barriers that callers already use
//                            around the producer/consumer phases.
//   cp_async_zfill        -> copy + zero-fill tail
// NOTE (Phase 1): if profiling shows the copy is on the critical path, revisit
// with an explicit register->LDS staging schedule per kernel.
// ============================================================================

#include <cuda_runtime.h>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    // AMD: LDS pointers are used directly; kept for API compatibility.
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

namespace detail {
template <int Bytes>
__device__ __forceinline__ void plain_copy(void* dst, const void* src) {
    if constexpr (Bytes == 16) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else if constexpr (Bytes == 8) {
        *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
    } else if constexpr (Bytes == 4) {
        *reinterpret_cast<unsigned*>(dst) = *reinterpret_cast<const unsigned*>(src);
    }
}
template <int Bytes>
__device__ __forceinline__ void copy_zfill(void* dst, const void* src, int src_bytes) {
    if (src_bytes >= Bytes) {
        plain_copy<Bytes>(dst, src);
    } else {
        // byte-granular copy of the valid head, zero the rest
        unsigned char* d = reinterpret_cast<unsigned char*>(dst);
        const unsigned char* s = reinterpret_cast<const unsigned char*>(src);
#pragma unroll
        for (int i = 0; i < Bytes; ++i) d[i] = (i < src_bytes) ? s[i] : 0;
    }
}
} // namespace detail

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    (void)Policy;
    detail::plain_copy<Bytes>(smem_dst, gmem_src);
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    (void)Policy;
    detail::copy_zfill<Bytes>(smem_dst, gmem_src, src_bytes);
}

__device__ __forceinline__ void cp_commit() {}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    detail::plain_copy<Bytes>(smem_dst, gmem_src);
}

__device__ __forceinline__ void pipe_commit() {}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
}

} // namespace ninfer::ops
