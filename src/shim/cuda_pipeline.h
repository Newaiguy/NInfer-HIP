#pragma once
// NInfer -> HIP port shim: cuda_pipeline.h
// NInfer only uses its own cp_async wrappers (ops/common/memory.cuh);
// any direct pipeline primitives map to the generic path.
#include <hip/hip_runtime.h>
