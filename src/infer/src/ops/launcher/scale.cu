#include "ninfer/ops/scale.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("scale_bf16: ") + what + ": " + cudaGetErrorString(status));
    }
}

__global__ void scale_kernel(__nv_bfloat16* data, std::size_t count, float factor) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    data[index] = __float2bfloat16(__bfloat162float(data[index]) * factor);
}

} // namespace

void scale_bf16(Tensor& x, float factor, cudaStream_t stream) {
    if (x.data == nullptr || x.dtype != DType::BF16) {
        throw std::runtime_error("scale_bf16: expected a materialized BF16 tensor");
    }
    const std::size_t count = x.bytes() / sizeof(std::uint16_t);
    if (count == 0) { return; }
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    scale_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), count, factor);
    check(cudaGetLastError(), "kernel launch");
}

} // namespace ninfer::ops
