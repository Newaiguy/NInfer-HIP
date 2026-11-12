// Generic BF16 x BF16 SIMT GEMM used when a shape has no hand-tuned kernel.
//
// The vision tower owns a handful of shapes (3456x1152, 4304x1152, 1152x1152, 1152x4304) that the
// specialized tables never covered, which used to force it onto 4 bit weights. This fallback keeps
// the same call shape as the tuned kernels so the dispatcher can hand any shape to it.

#include "ops/linear/bf16/bf16_launch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kTile = 32;
constexpr int kRowsPerThread = 4;
constexpr int kThreads = 256;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 value) {
    return __bfloat162float(value);
}

// Each block computes a 32x32 output tile: tokens [t0, t0+32) x outputs [n0, n0+32).
__global__ void bf16_generic_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ w,
                                         __nv_bfloat16* __restrict__ out, int tokens, int outputs,
                                         int inner) {
    __shared__ float x_tile[kTile][kTile + 1];
    __shared__ float w_tile[kTile][kTile + 1];

    const int t0 = blockIdx.y * kTile;
    const int n0 = blockIdx.x * kTile;
    const int lane = threadIdx.x;            // 0..255
    const int out_index = lane % kTile;      // output row inside the tile
    const int token_group = lane / kTile;    // 0..7, each owns 4 tokens

    float acc[kRowsPerThread] = {0.0F, 0.0F, 0.0F, 0.0F};

    for (int k0 = 0; k0 < inner; k0 += kTile) {
        // One row of x per thread: 256 threads cover the 32x32 tile exactly.
        const int row = lane / kTile;
        const int col = lane % kTile;
        for (int step = 0; step < kTile / 8; ++step) {
            const int t = t0 + row + step * 8;
            if (t < tokens && (k0 + col) < inner) {
                x_tile[row + step * 8][col] = bf16_to_float(x[static_cast<std::size_t>(t) * inner + k0 + col]);
            } else if ((k0 + col) < inner) {
                x_tile[row + step * 8][col] = 0.0F;
            }
            const int n = n0 + row + step * 8;
            if (n < outputs && (k0 + col) < inner) {
                w_tile[row + step * 8][col] = bf16_to_float(w[static_cast<std::size_t>(n) * inner + k0 + col]);
            } else if ((k0 + col) < inner) {
                w_tile[row + step * 8][col] = 0.0F;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const float weight = w_tile[out_index][kk];
#pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                acc[r] = fmaf(weight, x_tile[token_group * kRowsPerThread + r][kk], acc[r]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int t = t0 + token_group * kRowsPerThread + r;
        const int n = n0 + out_index;
        if (t < tokens && n < outputs) {
            out[static_cast<std::size_t>(t) * outputs + n] = __float2bfloat16(acc[r]);
        }
    }
}

} // namespace

void bf16_generic_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int inner = weight.k;
    const int outputs = weight.n;
    const int tokens = x.ne[1];
    const void* raw_weight = weight.payload != nullptr ? weight.payload : weight.qdata;
    const dim3 grid(static_cast<unsigned>((outputs + kTile - 1) / kTile),
                    static_cast<unsigned>((tokens + kTile - 1) / kTile));
    bf16_generic_gemm_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(raw_weight),
        static_cast<__nv_bfloat16*>(out.data), tokens, outputs, inner);
}

} // namespace ninfer::ops::detail
