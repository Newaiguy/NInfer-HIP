// B20 P0-2 triage: find the first layer/op whose output x contains NaN during the
// long-context prefill (the int8 collapse). Per-tag first-trigger records in a
// hipMalloc'ed global buffer (NO printf -> no TDR risk). Host reads the buffer back
// after the run with a plain hipMemcpy (no symbol mechanism).
#include "ninfer/ops/common/nan_triage_probe.h"

#include <cuda_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace ninfer {

namespace {
constexpr int kMaxTags = 1024;
// Host-held device pointers for the triage buffers (allocated once).
int* g_marks = nullptr;
NanRecord* g_records = nullptr;
} // namespace

// B20 P0-2: max |K scale| seen by the int8 decode QK path (fp16 precision check).
// Defined HERE (single TU with the reader); decode_t1_wmma.cuh externs it.
__device__ float ninfer_kv_scale_absmax = 0.0f;

__global__ void ninfer_nan_scan_kernel(const __nv_bfloat16* __restrict__ x, int rows,
                                       int cols, int tag, int* __restrict__ marks,
                                       NanRecord* __restrict__ records) {
    const int idx   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x);
    const int total = rows * cols;
    if (idx >= total || tag < 0 || tag >= kMaxTags) return;
    const float v = static_cast<float>(x[idx]);
    const bool bad = isnan(v) || isinf(v);
    const bool huge = !bad && fabsf(v) > 1.0e4f;
    if (bad || huge) {
        const int old = atomicCAS(&marks[tag], 0, 1);
        if (old == 0) {
            records[tag] = NanRecord{tag, idx / cols, idx - (idx / cols) * cols, v};
        }
    }
}

void ninfer_nan_scan(const void* x_bytes, int rows, int cols, int tag, void* stream) {
    if (g_marks == nullptr) {
        hipMalloc(&g_marks, sizeof(int) * kMaxTags);
        hipMalloc(&g_records, sizeof(NanRecord) * kMaxTags);
        hipMemset(g_marks, 0, sizeof(int) * kMaxTags);
        hipMemset(g_records, 0, sizeof(NanRecord) * kMaxTags);
    }
    const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(x_bytes);
    const int total        = rows * cols;
    ninfer_nan_scan_kernel<<<(total + 1023) / 1024, 1024, 0, static_cast<hipStream_t>(stream)>>>(
        x, rows, cols, tag, g_marks, g_records);
}

// Per-layer NaN counter: reduces the number of NaN/Inf elements in x and returns it
// to the host (synchronous). Also tracks max|finite v| — the int8 collapse shows a
// clean (NaN-free) x whose SQUARES overflow fp32 in rmsnorm, so the max magnitude is
// the actual smoking gun.
__global__ void ninfer_nan_count_kernel(const __nv_bfloat16* __restrict__ x, int total,
                                        unsigned int* __restrict__ out,
                                        float* __restrict__ maxabs) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (idx >= total) return;
    const float v = static_cast<float>(x[idx]);
    if (isnan(v) || isinf(v)) {
        atomicAdd(out, 1u);
        return;
    }
    const float a = fabsf(v);
    float cur = *maxabs;
    while (a > cur) {
        const float prev =
            atomicCAS(reinterpret_cast<int*>(maxabs), __float_as_int(cur), __float_as_int(a));
        if (prev == __float_as_int(cur)) { break; }
        cur = __int_as_float(prev);
    }
}

float ninfer_nan_count_last_maxabs = 0.0f;

int ninfer_nan_count(const void* x_bytes, int rows, int cols, void* stream) {
    static unsigned int* d_count = nullptr;
    static float* d_max = nullptr;
    if (d_count == nullptr) {
        const hipError_t em = hipMalloc(&d_count, sizeof(unsigned int));
        const hipError_t ex = hipMalloc(&d_max, sizeof(float));
        if (em != hipSuccess || ex != hipSuccess) { return -1000000; }
    }
    const hipError_t ez = hipMemset(d_count, 0, sizeof(unsigned int));
    const hipError_t ezm = hipMemset(d_max, 0, sizeof(float));
    if (ez != hipSuccess || ezm != hipSuccess) { return -2000000 + static_cast<int>(ez); }
    const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(x_bytes);
    const int total        = rows * cols;
    ninfer_nan_count_kernel<<<(total + 1023) / 1024, 1024, 0,
                              static_cast<hipStream_t>(stream)>>>(x, total, d_count, d_max);
    const hipError_t el = hipGetLastError();
    if (el != hipSuccess) { return -3000000 + static_cast<int>(el); }
    unsigned int h = 0;
    float mx = 0.0f;
    const hipError_t ec =
        hipMemcpy(&h, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost);
    const hipError_t em2 = hipMemcpy(&mx, d_max, sizeof(float), hipMemcpyDeviceToHost);
    if (ec != hipSuccess || em2 != hipSuccess) { return -4000000 + static_cast<int>(ec); }
    ninfer_nan_count_last_maxabs = mx;
    return static_cast<int>(h);
}

// Called once at the END of the run: prints every tag that ever triggered.
void ninfer_nan_triage_report() {    std::fprintf(stderr, "[nan-triage] per-tag first triggers:\n");
    if (g_marks == nullptr) {
        std::fprintf(stderr, "[nan-triage] no scans ran\n");
        std::fflush(stderr);
        return;
    }
    // Must sync the stream used by the scans before reading host-side.
    hipDeviceSynchronize();
    std::vector<int> marks(kMaxTags);
    std::vector<NanRecord> records(kMaxTags);
    const hipError_t e0 =
        hipMemcpy(marks.data(), g_marks, sizeof(int) * kMaxTags, hipMemcpyDeviceToHost);
    const hipError_t e1 =
        hipMemcpy(records.data(), g_records, sizeof(NanRecord) * kMaxTags, hipMemcpyDeviceToHost);
    if (e0 != hipSuccess || e1 != hipSuccess) {
        std::fprintf(stderr, "[nan-triage] copy failed: %s / %s\n", hipGetErrorString(e0),
                     hipGetErrorString(e1));
        std::fflush(stderr);
        return;
    }
    int count = 0;
    for (int tag = 0; tag < kMaxTags; ++tag) {
        if (marks[tag] != 0) {
            ++count;
            const NanRecord& r = records[tag];
            std::fprintf(stderr, "[nan-triage] tag=%d row=%d col=%d v=%f\n", r.tag, r.row, r.col,
                         r.value);
        }
    }
    if (count == 0) { std::fprintf(stderr, "[nan-triage] (no triggers)\n"); }
    // B20 P0-2: report the max |K scale| tracked by the decode kernel (fp16 check).
    float scale_max = 0.0f;
    const hipError_t e2 =
        hipMemcpyFromSymbol(&scale_max, ninfer_kv_scale_absmax, sizeof(float), 0,
                            hipMemcpyDeviceToHost);
    if (e2 == hipSuccess) {
        std::fprintf(stderr, "[nan-triage] decode K scale absmax=%f (fp16 prec breaks >~2048)\n",
                     scale_max);
    } else {
        std::fprintf(stderr, "[nan-triage] scale read failed: %s\n", hipGetErrorString(e2));
    }
    std::fflush(stderr);
}

} // namespace ninfer