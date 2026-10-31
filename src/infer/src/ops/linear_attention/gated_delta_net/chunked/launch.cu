#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <new>
#include <windows.h> // AMD port: QPC for NINFER_PROFILE_GDN stage timer

namespace ninfer::ops::detail::gated_delta_net {
std::size_t chunked_workspace_bytes(std::int32_t value_heads, std::int32_t tokens) {
    if (tokens <= 0) { return 0; }
    return chunked::workspace_bytes(value_heads, tokens);
}

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream) {
    const auto layout = chunked::compute_workspace_layout(v.ne[1], q.ne[2]);
    if (workspace == nullptr || workspace_bytes < layout.total_bytes) { throw std::bad_alloc(); }

    const DeviceSpan backing{workspace, workspace_bytes};
    const Tensor g_cumsum = layout.g_cumsum.bind(backing);
    const Tensor W        = layout.W.bind(backing);
    const Tensor U        = layout.U.bind(backing);
    const Tensor v_new    = layout.v_new.bind(backing);
    const Tensor h_chunk  = layout.h_chunk.bind(backing);

    chunked::prepare_wy_wu_config prepare{};
    prepare.H_qk         = q.ne[1];
    prepare.H_v          = v.ne[1];
    prepare.L            = q.ne[2];
    prepare.k            = static_cast<const __nv_bfloat16*>(k.data);
    prepare.v            = static_cast<const __nv_bfloat16*>(v.data);
    prepare.g_in         = static_cast<const float*>(g.data);
    prepare.beta         = static_cast<const float*>(beta.data);
    prepare.W            = static_cast<__nv_bfloat16*>(W.data);
    prepare.U            = static_cast<__nv_bfloat16*>(U.data);
    prepare.g_cumsum_out = static_cast<float*>(g_cumsum.data);
    prepare.stream       = stream;
    // AMD port: NINFER_PROFILE_GDN=1 times the three chunked stages.
    static const bool stage_timing = [] {
        const char* e = std::getenv("NINFER_PROFILE_GDN");
        return e != nullptr && e[0] == '1';
    }();
    LARGE_INTEGER chunkfreq = {}, chunktstart = {}, chunktA = {}, chunktB = {}, chunktC = {};
    if (stage_timing) {
        QueryPerformanceFrequency(&chunkfreq);
        QueryPerformanceCounter(&chunktstart);
    }
    CUDA_CHECK(chunked::launch_prepare_wy_wu(prepare));
    if (stage_timing) { CUDA_CHECK(cudaDeviceSynchronize()); QueryPerformanceCounter(&chunktA); }

    chunked::state_passing_config state_cfg{};
    state_cfg.H_qk      = q.ne[1];
    state_cfg.H_v       = v.ne[1];
    state_cfg.L         = q.ne[2];
    state_cfg.W         = static_cast<const __nv_bfloat16*>(W.data);
    state_cfg.U         = static_cast<const __nv_bfloat16*>(U.data);
    state_cfg.k         = static_cast<const __nv_bfloat16*>(k.data);
    state_cfg.g_cumsum  = static_cast<const float*>(g_cumsum.data);
    state_cfg.state_in  = static_cast<const float*>(ssm_state_in.data);
    state_cfg.v_new     = static_cast<__nv_bfloat16*>(v_new.data);
    state_cfg.h_chunk   = static_cast<__nv_bfloat16*>(h_chunk.data);
    state_cfg.state_out = static_cast<float*>(ssm_state_out.data);
    state_cfg.stream    = stream;
    CUDA_CHECK(chunked::launch_state_passing(state_cfg));
    if (stage_timing) { CUDA_CHECK(cudaDeviceSynchronize()); QueryPerformanceCounter(&chunktB); }

    chunked::chunk_output_config output{};
    output.H_qk     = q.ne[1];
    output.H_v      = v.ne[1];
    output.L        = q.ne[2];
    output.q        = static_cast<const __nv_bfloat16*>(q.data);
    output.k        = static_cast<const __nv_bfloat16*>(k.data);
    output.v_new    = static_cast<const __nv_bfloat16*>(v_new.data);
    output.g_cumsum = static_cast<const float*>(g_cumsum.data);
    output.h_chunk  = static_cast<const __nv_bfloat16*>(h_chunk.data);
    output.attn_out = static_cast<__nv_bfloat16*>(out.data);
    output.scale    = scale;
    output.stream   = stream;
    CUDA_CHECK(chunked::launch_output(output));
    if (stage_timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        QueryPerformanceCounter(&chunktC);
        static long calls = 0;
        static double a_ms = 0.0, b_ms = 0.0, c_ms = 0.0;
        a_ms += static_cast<double>(chunktA.QuadPart - chunktstart.QuadPart) * 1000.0 / static_cast<double>(chunkfreq.QuadPart);
        b_ms += static_cast<double>(chunktB.QuadPart - chunktA.QuadPart) * 1000.0 / static_cast<double>(chunkfreq.QuadPart);
        c_ms += static_cast<double>(chunktC.QuadPart - chunktB.QuadPart) * 1000.0 / static_cast<double>(chunkfreq.QuadPart);
        if (++calls % 48 == 0) {
            std::fprintf(stderr, "[gdn-chunked] %ld calls: prepare=%.1f state_pass=%.1f output=%.1f ms (per call %.3f/%.3f/%.3f)\n",
                         calls, a_ms, b_ms, c_ms, a_ms / 48.0, b_ms / 48.0, c_ms / 48.0);
            std::fflush(stderr);
            a_ms = b_ms = c_ms = 0.0;
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net
