#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "ninfer/ops/common/nan_triage_probe.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/gguf/gguf_linear.h" // AMD port: GGUF dual-weight SwiGLU workspace

#include <stdexcept>
#include <windows.h> // AMD port: QPC for NINFER_PROFILE_VERIFY MLP split

namespace ninfer::models::qwen3_5::execution {

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (is_gguf(gu.qtype)) {
        // GGUF SwiGLU: mlp/gate and mlp/up are separate block parents. activation
        // [gu.n, T] from the gate rows, then the residual-add down projection. The
        // dual-weight linear_swiglu/linear_add GGUF routes size their own q8_1 workspace.
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            const ops::detail::GgufShape up_shape{p.up.weight.qtype, p.up.weight.n, p.up.weight.k};
            (void)layout.alloc_bytes(ops::detail::gguf_swiglu_workspace_bytes(
                {gu.qtype, gu.n, gu.k}, &up_shape, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
        return layout.peak_bytes(1);
    }
    WorkspaceLayoutBuilder layout;
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        stream);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (is_gguf(gu.qtype)) {
        // GGUF SwiGLU: mlp/gate and mlp/up are separate block parents, so gate and up rows
        // both occupy gu.n rows and the dual-weight route projects them independently.
        // (No MTP on GGUF artifacts.)
        Tensor activation = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear_swiglu(hidden, gu, p.up.weight, activation, p.gate_up.policy, workspace,
                               stream);
        }
        if (std::getenv("NINFER_FFN_ACT_PROBE") != nullptr) {
            static int dump_taken = 0;
            const bool take = dump_taken == 0;
            if (take) { ++dump_taken; }
            const std::size_t n = static_cast<std::size_t>(gu.n) * columns;
            std::vector<std::uint16_t> av(n);
            const std::size_t hn = static_cast<std::size_t>(hidden.ne[0]) * hidden.ne[1];
            std::vector<std::uint16_t> hv(hn);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(av.data(), activation.data, n * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hv.data(), hidden.data, hn * 2, cudaMemcpyDeviceToHost));
            if (take) {
                std::FILE* af = std::fopen("D:/ninfer-hip-migrate/build/ffn_act.bin", "wb");
                if (af) { std::fwrite(av.data(), 2, n, af); std::fclose(af); }
                std::FILE* hf = std::fopen("D:/ninfer-hip-migrate/build/ffn_hin.bin", "wb");
                if (hf) { std::fwrite(hv.data(), 2, hn, hf); std::fclose(hf); }
            }
            std::fprintf(stderr, "[ffnact] gu.n=%d T=%d take=%d dumped %zu hidden=%zu\n", gu.n,
                         columns, take ? 1 : 0, n, hn);
            std::fflush(stderr);
        }
        ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        return;
    }
    if (mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        // AMD port: NINFER_PROFILE_VERIFY=1 splits the MTP MLP into its three ops.
        static const bool ffn_split = [] {
            const char* e = std::getenv("NINFER_PROFILE_VERIFY");
            return e != nullptr && e[0] == '1';
        }();
        struct FfnSplitTimer {
            bool on;
            double gate_up = 0.0, act = 0.0, down = 0.0;
            long calls = 0;
            LARGE_INTEGER f = {}, t0 = {}, t1 = {}, t2 = {}, t3 = {};
            void mark(double& acc) {
                CUDA_CHECK(cudaDeviceSynchronize());
                LARGE_INTEGER now;
                QueryPerformanceCounter(&now);
                acc += static_cast<double>(now.QuadPart - t0.QuadPart) * 1000.0 /
                       static_cast<double>(f.QuadPart);
                t0 = now;
            }
        };
        static FfnSplitTimer fs = [] {
            FfnSplitTimer v;
            v.on = ffn_split;
            QueryPerformanceFrequency(&v.f);
            return v;
        }();
        if (fs.on) { QueryPerformanceCounter(&fs.t0); }
{
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        if (fs.on) { fs.mark(fs.gate_up); }
        if (std::getenv("NINFER_NAN_TRIAGE") != nullptr) {
            ninfer::ninfer_nan_scan(gate_up.data, gu.n, static_cast<int>(gate_up.ne[1]), 700,
                                    stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, 0, gu.n / 2),
                      activation, stream);
        if (std::getenv("NINFER_NAN_TRIAGE") != nullptr) {
            ninfer::ninfer_nan_scan(activation.data, gu.n / 2, columns, 701, stream);
        }
        if (fs.on) { fs.mark(fs.act); }
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        if (std::getenv("NINFER_NAN_TRIAGE") != nullptr) {
            ninfer::ninfer_nan_scan(delta.data, down.n, columns, 702, stream);
        }
        if (fs.on) { fs.mark(fs.down); }
        ops::residual_add(delta, residual, stream);
        if (fs.on) {
            if (++fs.calls % 64 == 0) {
                std::fprintf(stderr,
                             "[ffn-split] %ld layers: gate_up=%.1f silu=%.1f down=%.1f ms "
                             "(per 64 layers)\n",
                             fs.calls, fs.gate_up, fs.act, fs.down);
                std::fflush(stderr);
                fs.gate_up = fs.act = fs.down = 0.0;
            }
        }
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream);
    }
    if (std::getenv("NINFER_NAN_TRIAGE") != nullptr) {
        ninfer::ninfer_nan_scan(activation.data, gu.n / 2, columns, 710, stream);
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
    if (std::getenv("NINFER_NAN_TRIAGE") != nullptr) {
        ninfer::ninfer_nan_scan(residual.data, static_cast<int>(residual.ne[0]),
                                static_cast<int>(residual.ne[1]), 711, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
