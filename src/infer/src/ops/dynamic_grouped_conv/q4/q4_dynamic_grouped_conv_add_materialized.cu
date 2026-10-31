// AMD port: Q4 variant of the dynamic grouped conv add. The Bonsai converter stores the
// DFlash2 branch projections as Q4_G64 (unlike the upstream Q8), so the projection step
// materializes BF16 via the Q4 GEMV/ksplit path; the finish (convolution + residual) is
// identical to the Q8 path and is reused here.
#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q4/q4_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 5120, kGroups = 320;

__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    const int index = col * kRows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * kRows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * kRows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    finish_value(row, col, width, __bfloat162float(projected[index]),
                 col % width ? __bfloat162float(projected[index - kRows]) : 0.0f, base, delta,
                 residual);
}

void materialized(const Tensor& x, const Weight& weight, const Tensor& base, const Tensor& delta,
                  Tensor& residual, Tensor& projected, cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({kRows, tokens});
    if (tokens == 1) {
        launch_q4_gemv_r4_w1_direct(flat, weight, result, stream);
    } else if (x.ne[0] == 4096) {
        if (tokens <= 4) launch_q4_ksplit<5120, 4096, 4>(flat, weight, result, stream);
        else if (tokens <= 8) launch_q4_ksplit<5120, 4096, 8>(flat, weight, result, stream);
        else if (tokens <= 16) launch_q4_ksplit<5120, 4096, 16>(flat, weight, result, stream);
        else if (tokens <= 32) launch_q4_ksplit<5120, 4096, 32>(flat, weight, result, stream);
        else launch_q4_mma_r64_c128(flat, weight, result, stream);
    } else {
        if (tokens <= 4) launch_q4_ksplit<5120, 17408, 4>(flat, weight, result, stream);
        else if (tokens <= 8) launch_q4_ksplit<5120, 17408, 8>(flat, weight, result, stream);
        else if (tokens <= 16) launch_q4_ksplit<5120, 17408, 16>(flat, weight, result, stream);
        else if (tokens <= 32) launch_q4_ksplit<5120, 17408, 32>(flat, weight, result, stream);
        else launch_q4_mma_r64_c128(flat, weight, result, stream);
    }
    const dim3 grid((kRows + 255) / 256, tokens);
    finish_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(projected.data),
                                            static_cast<const __nv_bfloat16*>(base.data),
                                            static_cast<const __nv_bfloat16*>(delta.data),
                                            static_cast<__nv_bfloat16*>(residual.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void q4_dynamic_grouped_conv_add_materialized_launch(const Tensor& x, const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream) {
    materialized(x, weight, base_kernel, finish_delta, residual, projected, stream);
}
} // namespace ninfer::ops::detail
