#pragma once
#include <msvc_host_compat.h>
// NInfer -> HIP port shim: cuda_runtime.h + cuda_runtime_api.h + cuda.h
// MSVC host .cpp builds: ROCm's amd_detail headers use GNU __attribute__ on
// device declarations; MSVC cannot parse them. Host code never needs them.
#if defined(_MSC_VER) && !defined(__clang__)
#define __attribute__(x)
#endif
// Pure-host MSVC .cpp builds only need the host runtime API; hip/hip_runtime.h pulls
// amd_detail device declarations (device_library_decls.h uses _Float16 etc.) that
// MSVC cannot parse.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__HIPCC__)
#include <hip/hip_runtime_api.h>
#else
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif
#include "cuda_warp.h"

#define cudaStream_t hipStream_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaEvent_t hipEvent_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset2D hipMemset2D
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventRecord hipEventRecord
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGraphCreate hipGraphCreate
// CUDA 12 3-arg instantiate -> HIP's InstantiateWithFlags
inline __host__ hipError_t __shim_graph_instantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
#define cudaGraphInstantiate __shim_graph_instantiate

// CUDA 12 exec-update result info (HIP exposes error-node out-param instead).
// hipcc defines __HIPCC__ and includes the real hip headers, which already declare
// hipGraphExecUpdateResult; use those names when compiling under hipcc.
#if defined(__HIPCC__)
#define cudaGraphExecUpdateSuccess hipGraphExecUpdateSuccess
#define cudaGraphExecUpdateError hipGraphExecUpdateError
#define cudaGraphExecUpdateErrorFunctionChanged hipGraphExecUpdateErrorFunctionChanged
#define cudaGraphExecUpdateErrorParametersChanged hipGraphExecUpdateErrorParametersChanged
#define cudaGraphExecUpdateErrorNodeTypeChanged hipGraphExecUpdateErrorNodeTypeChanged
#define cudaGraphExecUpdateErrorTopologyChanged hipGraphExecUpdateErrorTopologyChanged
typedef hipGraphExecUpdateResult cudaGraphExecUpdateResult;
#else
enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorFunctionChanged = 2,
    cudaGraphExecUpdateErrorParametersChanged = 3,
    cudaGraphExecUpdateErrorNodeTypeChanged = 4,
    cudaGraphExecUpdateErrorTopologyChanged = 5,
};
#endif
typedef struct {
    cudaGraphExecUpdateResult result;
    void* errorNode;
} cudaGraphExecUpdateResultInfo;
inline __host__ hipError_t __shim_graph_exec_update(hipGraphExec_t exec, hipGraph_t graph, cudaGraphExecUpdateResultInfo* info) {
    hipGraphNode_t err_node = nullptr;
    hipGraphExecUpdateResult r;
    hipError_t e = hipGraphExecUpdate(exec, graph, &err_node, &r);
    info->result    = (e == hipSuccess) ? cudaGraphExecUpdateSuccess : cudaGraphExecUpdateError;
    info->errorNode = err_node;
    return e;
}
#define cudaGraphExecUpdate __shim_graph_exec_update
#define cudaGraphUpload hipGraphUpload
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams
#define cudaStreamGetCaptureInfo hipStreamGetCaptureInfo
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaMallocHost hipMallocHost
#define cudaFreeHost hipFreeHost
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
template <class F>
inline __host__ hipError_t __shim_func_set_attr(F* func, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}
#define cudaFuncSetAttribute(func, ...) __shim_func_set_attr(func, __VA_ARGS__)
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaLaunchKernelEx hipLaunchKernelEx
#define cudaLaunchConfig_t hipLaunchConfig_t
#define cudaLaunchAttribute hipLaunchAttribute
#define cudaLaunchAttributeCooperative hipLaunchAttributeCooperative
#define cudaLaunchAttributePriority hipLaunchAttributePriority

// generic->shared address conversion (operand for async-copy asm; semantics
// handled when porting memory.cuh, cast is enough to compile)
inline __device__ unsigned __cvta_generic_to_shared(const void* p) { return static_cast<unsigned>(reinterpret_cast<unsigned long long>(p)); }

#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventQuery hipEventQuery
#define cudaErrorNotReady hipErrorNotReady

// CUDA 12 kernel-parameter attribute; plain pass-by-value on HIP
#define __grid_constant__

// CUDA 12.4+ per-kernel register cap; no HIP equivalent on gfx12, ignore
#define __maxnreg__(n)
#define __NV_SATFINITE 0
#define __NV_E4M3 0
#define __NV_E5M2 1
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetLastErrorString hipGetLastErrorString

#define CUDA_CHECK(x) do { auto _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); abort(); } } while (0)

#define cudaAssert(cond, file, line) assert((cond) && (file) && (line))
