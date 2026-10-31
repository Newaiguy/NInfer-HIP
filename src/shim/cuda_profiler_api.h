#pragma once
// NInfer -> HIP port shim: profiler API stub (no-op)
extern "C" typedef int cudaProfilerStop_t;
inline void cudaProfilerStart() {}
inline void cudaProfilerStop() {}
