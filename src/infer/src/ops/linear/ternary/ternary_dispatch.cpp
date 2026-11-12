// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_dispatch.h"

#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy) {
    if (n <= 0 || k <= 0 || t <= 0) {
        throw std::invalid_argument("ternary linear: unsupported shape or T");
    }
    // The reference kernel decodes whole 128-weight groups, so K has to be a whole number
    // of them. Every width the model uses (5120, 6144, 10240, 17408) satisfies this, and
    // row_split_geometry() pads to the same 128 boundary, so no padding groups exist.
    if (k % PTQ1RowSplitStorage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    // Both ternary formats take the same schedule; the atom is chosen from w.qtype inside
    // the launch. Decode (T = 1) and small prefill use different token tiles.
    return t == 1 ? launch_ternary_gemm_t1 : launch_ternary_gemm_t8;
}

void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x_folded.ne[1], policy);
    launch(x_folded, w, out, out_row_stride, stream);
}

void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream) {
    ternary_dispatch_basis_strided(x_folded, w, out, w.n, policy, stream);
}

// AMD port: see ternary_dispatch.h. Thread-local so a host worker pool running several
// sequences cannot leak the mode across lanes.
thread_local bool t_ternary_verify_mode = false;

void set_ternary_verify_mode(bool enabled) { t_ternary_verify_mode = enabled; }

bool ternary_verify_mode() { return t_ternary_verify_mode; }

void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x.ne[1], policy);

    // B24 diagnosis: NINFER_SKIP_GEMV=1 replaces every ternary GEMV (rotation +
    // launch) with a memset so the wall-clock budget of "all GEMV work" can be
    // measured directly (paired with NINFER_SKIP_ATTN).
    static const bool skip_gemv = [] {
        const char* e = std::getenv("NINFER_SKIP_GEMV");
        return e != nullptr && e[0] == '1';
    }();
    if (skip_gemv) {
        hipMemsetAsync(out.data, 0,
                       static_cast<std::size_t>(w.n) * x.ne[1] * 2, stream);
        return;
    }

    if (!ternary_rotation_enabled()) {
        // NINFER_TERNARY_HADAMARD=0: run the GEMM against the raw activation so the full forward
        // pass (and therefore the M4 decode speed) is measurable. The result is numerically
        // meaningless -- the activation is in the wrong basis -- which is the point: it separates
        // "the ternary decode is broken" from "the rotation is broken" without a rebuild.
        launch(x, w, out, w.n, stream);
        return;
    }
    if (workspace == nullptr) {
        // Name the shape in the message: a folded ternary weight reaching the workspace-free
        // entry point is a call-site wiring gap, and the shape is what identifies the weight.
        throw std::invalid_argument(
            "ternary linear: the folded basis needs a rotation workspace [N=" +
            std::to_string(w.n) + ", K=" + std::to_string(w.k) + ", T=" +
            std::to_string(x.ne[1]) + ", qtype=" + std::to_string(static_cast<int>(w.qtype)) + "]");
    }

    // Scoped: the scratch is handed back when this op returns, so it does not accumulate across
    // the (many) graph constructions of one load. Sequential reuse on one stream is safe.
    // folded_activation() rejects a ternary weight with no sign block, so an unfolded artifact
    // fails loudly here instead of silently multiplying by unfolded weights.
    auto scope = workspace->scope();
    static const bool gpu_time = [] {
        const char* e = std::getenv("NINFER_TERN_GPU_TIME");
        return e != nullptr && e[0] == '1';
    }();
    if (gpu_time) {
        cudaStreamSynchronize(stream);
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, stream);
        const Tensor activation = folded_activation(x, w, *workspace, stream);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float rot_ms = 0.0f;
        cudaEventElapsedTime(&rot_ms, e0, e1);
        cudaEventRecord(e0, stream);
        launch(activation, w, out, w.n, stream);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float gemv_ms = 0.0f;
        cudaEventElapsedTime(&gemv_ms, e0, e1);
        static int tc = 0;
        if (++tc <= 800) {
            std::fprintf(stderr, "[tern-gpu] n=%d k=%d t=%d rot=%.4f gemv=%.4f ms\n", w.n, w.k,
                         x.ne[1], rot_ms, gemv_ms);
            std::fflush(stderr);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return;
    }
    const Tensor activation = folded_activation(x, w, *workspace, stream);
    launch(activation, w, out, w.n, stream);
}

} // namespace ninfer::ops::detail
