#include "models/qwen3_5/execution/gdn.h"

#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ops/linear/gguf/ggml_bridge.h"          // AMD port probe: engine-dequantized rows
#include "ops/linear/ternary/ternary_rotation.h" // AMD port: ternary

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

namespace {

// AMD port: folded ternary split parents need a [5120, T] rotation buffer from the workspace.
bool is_ternary_pair(const ops::PairedProjectionWeights& pair) noexcept {
    return pair.first.qtype == QType::PTQ1_0_G128 || pair.first.qtype == QType::PQ2_0_G128 ||
           pair.first.qtype == QType::T2_G128_FP16;
}

} // namespace

std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters, std::int32_t first,
                                           std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("GDN projection: invalid column interval");
    }
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return ops::gdn_input_proj_workspace_capacity_bytes(*gguf, first, last);
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        return is_ternary_pair(*pair) ? ops::detail::ternary_rotation_workspace_bytes(5120, last)
                                      : 0;
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& w = single->weight;
        return ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, single->policy,
                                                            first, last);
    }
    return 0;
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                         std::int32_t batch, std::int32_t first_width,
                                         std::int32_t last_width) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return std::max(std::size_t{1},
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            *gguf, batch, first_width, last_width));
    }
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    // The native overlap contract takes a disjoint span even for a zero-scratch fused route.
    return std::max(std::size_t{1}, bytes);
}

std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                       std::int32_t batch, std::int32_t first_width,
                                       std::int32_t last_width) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return std::max(std::size_t{1},
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            *gguf, batch, first_width, last_width));
    }
    const auto* single = std::get_if<LinearParameters>(&parameters.projection);
    std::size_t bytes;
    if (single && single->weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single->weight;
        bytes         = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single->policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    return std::max(std::size_t{1}, bytes);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj(hidden, *gguf, qkv, z, workspace, stream);
        if (std::getenv("NINFER_GDN_PROJ_PROBE") != nullptr) {
            // Dump only the first GDN layer's values (layer0); every GDN layer calls this
            // probe and would otherwise overwrite the same files.
            static int dump_taken = 0;
            const bool take = dump_taken == 0;
            if (take) { ++dump_taken; }
            cudaStreamSynchronize(stream);
            std::uint16_t q0[4] = {};
            CUDA_CHECK(cudaMemcpy(q0, qkv.data, sizeof(q0), cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[gdnproj] qkv=[%d,%d] z=[%d,%d] q0=%u,%u,%u,%u\n",
                         qkv.ne[0], qkv.ne[1], z.ne[0], z.ne[1],
                         static_cast<unsigned>(q0[0]), static_cast<unsigned>(q0[1]),
                         static_cast<unsigned>(q0[2]), static_cast<unsigned>(q0[3]));
            if (take) {
            static FILE* hf = std::fopen("D:/ninfer-hip-migrate/build/hidden_t0.bin", "wb");
            if (hf) {
                std::vector<std::uint16_t> hv(static_cast<std::size_t>(hidden.ne[0]) *
                                              hidden.ne[1]);
                CUDA_CHECK(cudaMemcpy(hv.data(), hidden.data, hv.size() * 2,
                                      cudaMemcpyDeviceToHost));
                std::fwrite(hv.data(), 2, hv.size(), hf);
                std::fclose(hf);
                hf = nullptr;
            }
            std::fprintf(stderr, "[gdnproj] hidden dumped %d x %d\n", hidden.ne[0], hidden.ne[1]);
            std::vector<std::uint16_t> qv(static_cast<std::size_t>(qkv.ne[0]) * qkv.ne[1]);
            std::vector<std::uint16_t> zv(static_cast<std::size_t>(z.ne[0]) * z.ne[1]);
            CUDA_CHECK(cudaMemcpy(qv.data(), qkv.data, qv.size() * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(zv.data(), z.data, zv.size() * 2, cudaMemcpyDeviceToHost));
            std::FILE* qf = std::fopen("D:/ninfer-hip-migrate/build/gdn_qkv.bin", "wb");
            if (qf) { std::fwrite(qv.data(), 2, qv.size(), qf); std::fclose(qf); }
            std::FILE* zf = std::fopen("D:/ninfer-hip-migrate/build/gdn_z.bin", "wb");
            if (zf) { std::fwrite(zv.data(), 2, zv.size(), zf); std::fclose(zf); }
            std::fprintf(stderr, "[gdnproj] qkv=%dx%d z=%dx%d dumped\n", qkv.ne[0], qkv.ne[1],
                         z.ne[0], z.ne[1]);
            // Probe: engine-dequantized first query rows (FP32), vs the CPU reference.
            const auto& w0 = gguf->parts.front().weight;
            // AMD port diagnostic: dump the RAW quantized bytes the engine reads for the
            // first qkv rows, to compare against the artifact/GGUF bytes (row 0..3).
            {
                cudaStreamSynchronize(stream);
                const std::int64_t rb0 = (w0.k / 256) * 136; // IQ4_XS row bytes
                std::vector<std::uint8_t> wraw(static_cast<std::size_t>(4) * rb0);
                CUDA_CHECK(cudaMemcpy(wraw.data(), w0.qdata, wraw.size(), cudaMemcpyDeviceToHost));
                std::FILE* wf = std::fopen("D:/ninfer-hip-migrate/build/eng_qkv_rows.bin", "wb");
                if (wf) { std::fwrite(wraw.data(), 1, wraw.size(), wf); std::fclose(wf); }
                std::fprintf(stderr, "[gdnproj] qkv raw bytes dumped: %d bytes (4 rows x %lld rb)\n",
                             static_cast<int>(wraw.size()), static_cast<long long>(rb0));
                std::fflush(stderr);
            }
            const int dqk = static_cast<int>(w0.k);
            std::vector<float> dq(static_cast<std::size_t>(2) * dqk);
            const std::int64_t rb =
                std::int64_t(dqk / 256) * 136; // IQ4_XS block bytes
            ops::gguf::dequantize_rows(ops::gguf::GgmlType::IQ4_XS, w0.qdata, rb, dqk, nullptr, 2,
                                       dq.data(), dqk, stream);
            cudaStreamSynchronize(stream);
            std::FILE* df = std::fopen("D:/ninfer-hip-migrate/build/query_deq.bin", "wb");
            if (df) { std::fwrite(dq.data(), 4, dq.size(), df); std::fclose(df); }
            std::fprintf(stderr, "[gdnproj] query_deq dumped: qtype=%d n=%d k=%d first=%f,%f\n",
                         static_cast<int>(w0.qtype), static_cast<int>(w0.n), dqk,
                         static_cast<double>(dq[0]), static_cast<double>(dq[1]));
            std::fflush(stderr);
            std::fflush(stderr);
            }
        }
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        if (is_ternary_pair(*pair)) {
            // AMD port: folded ternary parents need the rotation scratch from the workspace.
            ops::gdn_input_proj(hidden, pair->first, pair->second, qkv, z, workspace, stream);
            return;
        }
        ops::gdn_input_proj(hidden, pair->first, pair->second, qkv, z, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj(hidden, single.weight, qkv, z, single.policy, workspace, stream);
    }
}

void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      WorkspaceArena& workspace, DeviceExecutionView execution) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.control)) {
        ops::gdn_norm_gating_proj(residual, norm, epsilon, pair->first, pair->second,
                                  parameters.a_log, parameters.dt_bias, workspace, hidden, g, beta,
                                  execution);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.control);
        ops::gdn_norm_gating_proj(residual, norm, epsilon, single.weight, parameters.a_log,
                                  parameters.dt_bias, workspace, hidden, g, beta, execution);
    }
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(gdn_snapshot_workspace_bytes(
        parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, *gguf, parameters.convolution, conv_states,
                                          valid_columns, initial_slots, destination_slots, query,
                                          key, value, z, scratch, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, pair->first, pair->second, parameters.convolution,
                                          conv_states, valid_columns, initial_slots,
                                          destination_slots, query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_snapshot(
            hidden, single.weight, parameters.convolution, conv_states, valid_columns,
            initial_slots, destination_slots, query, key, value, z, single.policy, scratch, stream);
    }
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    WorkspaceArena scratch(workspace.alloc_bytes(
        gdn_record_workspace_bytes(parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(hidden, *gguf, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, scratch, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(hidden, pair->first, pair->second, parameters.convolution,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, z, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_record(hidden, single.weight, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, single.policy, scratch, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
