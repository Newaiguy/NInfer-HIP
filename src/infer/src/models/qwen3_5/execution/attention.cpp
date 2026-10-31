#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rope.h"
#include "ops/linear/ternary/ternary_rotation.h" // AMD port: ternary

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return ops::attn_input_proj_workspace_capacity_bytes(*gguf, first, last);
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        // AMD port: folded ternary parents rotate the activation through the workspace first.
        if (pair->first.qtype == QType::PTQ1_0_G128 ||
        pair->first.qtype == QType::PQ2_0_G128 ||
            pair->first.qtype == QType::T2_G128_FP16) {
            return ops::detail::ternary_rotation_workspace_bytes(5120, last);
        }
        return 0;
    }
    const auto& single = std::get<LinearParameters>(parameters.projection);
    const auto& weight = single.weight;
    return ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                         single.policy, first, last);
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(hidden, *gguf, query, gate, key, value, workspace, stream);
        return;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        if (pair->first.qtype == QType::PTQ1_0_G128 ||
        pair->first.qtype == QType::PQ2_0_G128 ||
            pair->first.qtype == QType::T2_G128_FP16) {
            // AMD port: folded ternary parents need the rotation scratch from the workspace.
            ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value,
                                 workspace, stream);
            return;
        }
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(hidden, single.weight, query, gate, key, value, single.policy,
                             workspace, stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, stream);
}

} // namespace ninfer::models::qwen3_5::execution
