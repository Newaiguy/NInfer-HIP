#pragma once
// NInfer -> HIP port shim: cooperative_groups.h
// Only grid-cooperative launch is referenced in the tree (grid_group::size /
// this_grid()). HIP exposes these via hipLaunchCooperativeKernel + grid params;
// provide a minimal compile-through surface used by bf16_gdn_gating_proj.
#include <hip/hip_runtime.h>

namespace cooperative_groups {

class grid_group {
public:
    __device__ unsigned size() const { return hipBlockDim_x * hipGridDim_x; }
    __device__ unsigned rank() const { return hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; }
    // NOTE: real grid sync requires hipLaunchCooperativeKernel; no-op for census
    __device__ void sync() const {}
    __device__ void synchronize() const {}
    __device__ bool is_valid() const { return true; }
};

__device__ inline grid_group this_grid() { return grid_group{}; }

class thread_block {
public:
    __device__ unsigned size() const { return hipBlockDim_x; }
    __device__ unsigned thread_rank() const { return hipThreadIdx_x; }
    __device__ void sync() const { __syncthreads(); }
};

__device__ inline thread_block this_thread_block() { return thread_block{}; }

} // namespace cooperative_groups
