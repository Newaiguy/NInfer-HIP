#pragma once

// B20 P0-2 triage helper (ops/common/nan_triage_probe.cu): device scan of a BF16
// tensor for NaN/Inf, writing records into a global buffer (no printf). Call
// ninfer_nan_scan after the ops you want to check; call ninfer_nan_triage_report()
// once at the end of the run to print the first records. Gated by NINFER_NAN_TRIAGE
// in the caller.
namespace ninfer {
struct NanRecord {
    int tag;
    int row;
    int col;
    float value;
};
void ninfer_nan_scan(const void* x, int rows, int cols, int tag, void* stream);
void ninfer_nan_triage_report();
int ninfer_nan_count(const void* x, int rows, int cols, void* stream);
extern float ninfer_nan_count_last_maxabs;
} // namespace ninfer