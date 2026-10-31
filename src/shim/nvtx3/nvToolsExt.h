#pragma once
// NInfer -> HIP port shim: NVTX profiling -> no-op stub.
// NInfer's nvtx.h only needs types + function symbols for push/pop ranges;
// on Windows/NVIDIA-hostless builds there is no profiler attached, so all
// calls compile away. Types sized to keep struct layout warnings away.
#ifndef NVTX_VERSION
#define NVTX_VERSION 1
#endif
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE 64
#define NVTX_COLOR_ARGB 1
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 3
#define NVTX_MESSAGE_TYPE_REGISTERED 5

typedef void* nvtxDomainHandle_t;
typedef void* nvtxStringHandle_t;
typedef unsigned long long nvtxRangeId_t;

typedef union {
    const char*           ascii;
    const wchar_t*        unicode;
    nvtxStringHandle_t    registered;
} nvtxMessageValue_t;

typedef union {
    unsigned long long ullValue;
    long long          llValue;
    double             dValue;
} nvtxPayloadValue_t;

typedef struct {
    unsigned short     version;
    unsigned short     size;
    unsigned           category;
    int                colorType;
    unsigned           color;
    nvtxMessageValue_t message;
    unsigned           messageType;
    unsigned           payloadType;
    nvtxPayloadValue_t payload;
    int                reserved0;
} nvtxEventAttributes_t;

inline nvtxDomainHandle_t nvtxDomainCreateA(const char*) { return nullptr; }
inline void nvtxDomainNameCategoryA(nvtxDomainHandle_t, std::uint32_t, const char*) {}
inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t, const char*) { return nullptr; }
inline int nvtxDomainRangePushEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) { return 0; }
inline int nvtxDomainRangePop(nvtxDomainHandle_t) { return 0; }
inline nvtxRangeId_t nvtxDomainRangeStartEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) { return 0; }
inline void nvtxDomainRangeEnd(nvtxDomainHandle_t, nvtxRangeId_t) {}
