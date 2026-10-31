#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool gguf_parent(const WeightInput& input) { return gguf_input(input); }

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

const std::vector<std::uint64_t>& matrix(const WeightInput& input) {
    require(input.weight.shape.size() == 2, "projection weight must be a matrix");
    const auto elements = weight_element_count(input.weight.shape);
    require(!input.weight.parts.empty(), "projection weight has no parent regions");
    std::uint64_t covered = 0;
    for (const auto& part : input.weight.parts) {
        require(part.parent && part.begin < part.end &&
                    part.end <= part.parent->geometry.elements &&
                    part.end - part.begin <= elements - covered,
                "projection region exceeds its parent or logical matrix");
        covered += part.end - part.begin;
    }
    require(covered == elements, "projection regions do not cover its logical matrix");
    return input.weight.shape;
}

WeightView concatenate_rows(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    const auto columns = matrix(inputs.front())[1];
    WeightView out{{0, columns}, {}};
    for (const auto& input : inputs) {
        const auto& shape = matrix(input);
        require(shape[1] == columns &&
                    shape[0] <= std::numeric_limits<std::uint64_t>::max() - out.shape[0],
                "projection bank has inconsistent columns or overflowing rows");
        out.shape[0] += shape[0];
        out.parts.insert(out.parts.end(), input.weight.parts.begin(), input.weight.parts.end());
    }
    return out;
}

bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    auto end           = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (!parent || part.parent != parent || part.begin != end || part.end <= part.begin) {
            return false;
        }
        end = part.end;
    }
    return true;
}

LinearPolicy common_policy(std::span<const WeightInput> inputs) {
    auto result = LinearPolicy::AllowA4;
    for (const auto& input : inputs) {
        switch (input.policy) {
        case LinearPolicy::A16Only:
            result = LinearPolicy::A16Only;
            break;
        case LinearPolicy::AllowA8:
            if (result == LinearPolicy::AllowA4) { result = LinearPolicy::AllowA8; }
            break;
        case LinearPolicy::AllowA4:
            break;
        default:
            throw std::invalid_argument("invalid projection activation policy");
        }
    }
    return result;
}

SingleProjectionWeight single(std::span<const WeightInput> inputs) {
    auto view         = concatenate_rows(inputs);
    const auto region = contiguous_weight_region(view);
    const auto policy = common_policy(inputs);
    float divisor     = 0;
    if (region.parent->geometry.format == QType::NVFP4) {
        require(std::isfinite(region.parent->weight_scale_divisor) &&
                    region.parent->weight_scale_divisor > 0,
                "NVFP4 parent requires a positive weight divisor");
        for (const auto& input : inputs) {
            if (!input.activation_input_divisor) {
                require(!allows_a4(policy), "NVFP4 A4 native input requires an activation divisor");
                continue;
            }
            require(std::isfinite(*input.activation_input_divisor) &&
                        *input.activation_input_divisor > 0,
                    "NVFP4 native input requires a positive activation divisor");
            if (divisor == 0) {
                // A16 does not read this auxiliary; retain a stored positive value for the ABI.
                divisor = *input.activation_input_divisor;
            } else if (allows_a4(policy)) {
                // Current NVFP4 consumers use A16 for AllowA8 as well. Only their A4 route
                // quantizes the shared activation and therefore requires a common divisor.
                require(std::bit_cast<std::uint32_t>(divisor) ==
                            std::bit_cast<std::uint32_t>(*input.activation_input_divisor),
                        "shared NVFP4 native input requires identical activation divisors");
            }
        }
        // The legacy native ABI validates this field even on A16, where it is never read.
        // Keep that ABI detail out of the artifact's optional Use auxiliaries.
        if (divisor == 0) { divisor = 1.0F; }
    }
    auto result = native_weight(view, divisor, inputs.front().folded_signs);
    // Preserve the optional GGUF input gather through the single-projection path. This is
    // required by GDN output, whose stored value-head columns remain in tiled order.
    result.input_columns = inputs.front().input_columns.data != nullptr
                               ? static_cast<const std::int32_t*>(inputs.front().input_columns.data)
                               : nullptr;
    return {result, policy};
}

// GGUF parts of a fused input projection, merged where consecutive logical inputs continue one
// parent's rows into one output's rows. GGUF models are dense and unrotated, so no Hadamard
// signs travel with them.
GgufProjectionWeights gguf_projection(std::span<const WeightInput> inputs,
                                      std::span<const std::int32_t> outputs,
                                      std::span<const std::int32_t> rows) {
    GgufProjectionWeights result;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        require(gguf_parent(inputs[i]), "input projection: GGUF parts cannot mix formats");
        auto part = single(inputs.subspan(i, 1));
        if (!result.parts.empty()) {
            auto& last = result.parts.back();
            const auto block = gguf_block_shape(last.weight.qtype);
            const std::int64_t row_bytes =
                std::int64_t(last.weight.k / block.elements) * block.bytes;
            const auto* end =
                static_cast<const std::byte*>(last.weight.qdata) + last.weight.n * row_bytes;
            if (part.weight.qtype == last.weight.qtype && part.weight.qdata == end &&
                outputs[i] == last.output && rows[i] == last.row + last.weight.n &&
                inputs[i].input_columns.data == last.input_columns) {
                last.weight.n += part.weight.n;
                last.weight.shape[0] = last.weight.padded_shape[0] = last.weight.n;
                continue;
            }
        }
        const auto* columns =
            inputs[i].input_columns.data != nullptr
                ? static_cast<const std::int32_t*>(inputs[i].input_columns.data)
                : nullptr;
        result.parts.push_back({part.weight, outputs[i], rows[i], columns});
    }
    return result;
}

ProjectionWeights input_projection(std::span<const WeightInput, 4> inputs, bool attention) {
    const auto& q      = matrix(inputs[0]);
    const auto& k      = matrix(inputs[1]);
    const auto& third  = matrix(inputs[2]);
    const auto& fourth = matrix(inputs[3]);
    const bool dense =
        attention ? q == std::vector<std::uint64_t>{6144, 5120} &&
                        k == std::vector<std::uint64_t>{1024, 5120} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 5120} && k == q &&
                        third == std::vector<std::uint64_t>{6144, 5120} && fourth == third;
    const bool moe =
        attention ? q == std::vector<std::uint64_t>{4096, 2048} &&
                        k == std::vector<std::uint64_t>{512, 2048} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 2048} && k == q &&
                        third == std::vector<std::uint64_t>{4096, 2048} && fourth == third;
    require(dense || moe, "input projection: unsupported logical projection geometry");
    if (std::any_of(inputs.begin(), inputs.end(), gguf_parent)) {
        require(dense, "input projection: GGUF parts need the dense geometry");
        if (attention) {
            const std::array<std::int32_t, 4> outputs{0, 2, 1, 3};
            const std::array<std::int32_t, 4> rows{0, 0, 0, 0};
            return gguf_projection(inputs, outputs, rows);
        }
        const auto& qshape = matrix(inputs[0]);
        const auto& kshape = matrix(inputs[1]);
        const std::array<std::int32_t, 4> outputs{0, 0, 0, 1};
        const std::array<std::int32_t, 4> rows{
            0, static_cast<std::int32_t>(qshape[0]),
            static_cast<std::int32_t>(qshape[0] + kshape[0]), 0};
        return gguf_projection(inputs, outputs, rows);
    }
    const auto joined = concatenate_rows(inputs);
    if (contiguous(joined)) {
        auto result       = single(inputs);
        const auto format = result.weight.qtype;
        const bool supported =
            (moe && format == QType::Q8_G32_FP16) ||
            (dense && (format == QType::NVFP4 || format == QType::FP8_E4M3FN_ROW_BF16 ||
                       (attention && format == QType::BF16)));
        require(supported, "input projection: unsupported single-parent format");
        return result;
    }
    require(dense, "input projection: unsupported multi-parent geometry");
    const auto first  = single(inputs.first<2>());
    const auto second = single(inputs.last<2>());
    // AMD port: folded (rotated-basis) ternary split parents keep the two-object form; each
    // half is a complete RowSplit parent carrying its own sign block for the shared width.
    const bool ternary_pair =
        ((first.weight.qtype == QType::PTQ1_0_G128 || first.weight.qtype == QType::PQ2_0_G128 ||
          first.weight.qtype == QType::T2_G128_FP16) &&
         first.weight.qtype == second.weight.qtype);
    require((first.weight.qtype == QType::Q4_G64_FP16 &&
             second.weight.qtype == QType::Q5_G64_FP16) ||
                ternary_pair,
            "input projection: paired native form requires Q4 and Q5");
    return PairedProjectionWeights{first.weight, second.weight};
}

} // namespace

// Whether the logical matrix is rows of a stored GGUF block parent (GgufBlocks layout).
bool gguf_input(const WeightInput& input) {
    return !input.weight.parts.empty() && input.weight.parts.front().parent != nullptr &&
           is_gguf(input.weight.parts.front().parent->geometry.format);
}

SingleProjectionWeight prepare_linear_weight(const WeightInput& input) {
    return single({&input, 1});
}

SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows) {
    return single(rows);
}

SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                       const WeightInput& key,
                                                       const WeightInput& value) {
    const auto& q = matrix(query);
    require((q == std::vector<std::uint64_t>{4096, 2048} ||
             q == std::vector<std::uint64_t>{4096, 5120}) &&
                matrix(key) == std::vector<std::uint64_t>{1024, q[1]} &&
                matrix(value) == matrix(key),
            "QKV input projection: unsupported logical geometry");
    const std::array inputs{query, key, value};
    auto result = single(inputs);
    require(result.weight.qtype == QType::Q8_G32_FP16,
            "QKV input projection: native form requires Q8");
    return result;
}

ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                  const WeightInput& gate,
                                                  const WeightInput& value) {
    const std::array inputs{query, key, gate, value};
    return input_projection(inputs, true);
}

ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                 const WeightInput& value, const WeightInput& z) {
    const std::array inputs{query, key, value, z};
    return input_projection(inputs, false);
}

ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a, const WeightInput& b) {
    const auto& shape = matrix(a);
    require(shape == matrix(b) && (shape == std::vector<std::uint64_t>{48, 5120} ||
                                   shape == std::vector<std::uint64_t>{32, 2048}),
            "GDN control: unsupported A/B geometry");
    const std::array inputs{a, b};
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        require(result.weight.qtype == QType::BF16, "GDN control requires BF16 weights");
        return result;
    }
    require(shape[0] == 48, "GDN control: this geometry requires a combined parent");
    const auto first  = prepare_linear_weight(a);
    const auto second = prepare_linear_weight(b);
    require(first.weight.qtype == QType::BF16 && second.weight.qtype == QType::BF16,
            "GDN control requires BF16 weights");
    return PairedProjectionWeights{first.weight, second.weight};
}

SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                    const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    return single(inputs);
}

SingleProjectionWeight prepare_gguf_swiglu_gate(const WeightInput& gate) {
    (void)matrix(gate);
    require(gguf_input(gate), "gguf SwiGLU gate must live in a GGUF block parent");
    const std::array inputs{gate};
    return single(inputs);
}

SingleProjectionWeight prepare_gguf_swiglu_up(const WeightInput& up) {
    (void)matrix(up);
    require(gguf_input(up), "gguf SwiGLU up must live in a GGUF block parent");
    const std::array inputs{up};
    return single(inputs);
}

SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down) {
    require(expert_gate_up.size() == 512 && expert_down.size() == 256,
            "SparseMoe native input requires 256 experts");
    require(matrix(router) == std::vector<std::uint64_t>{256, 2048} &&
                matrix(shared_score) == std::vector<std::uint64_t>{1, 2048},
            "SparseMoe router geometry differs");
    for (const auto& input : expert_gate_up) {
        require(matrix(input) == std::vector<std::uint64_t>{512, 2048},
                "SparseMoe gate/up geometry differs");
    }
    for (const auto& input : expert_down) {
        require(matrix(input) == std::vector<std::uint64_t>{2048, 512},
                "SparseMoe down geometry differs");
    }
    require(matrix(shared_gate) == std::vector<std::uint64_t>{512, 2048} &&
                matrix(shared_up) == matrix(shared_gate) &&
                matrix(shared_down) == std::vector<std::uint64_t>{2048, 512},
            "SparseMoe shared expert geometry differs");
    const std::array router_inputs{router, shared_score};
    const auto router_bank  = single(router_inputs);
    const auto gate_up_bank = single(expert_gate_up);
    const auto down_bank    = single(expert_down);
    const auto shared       = prepare_linear_swiglu_weight(shared_gate, shared_up);
    const auto down         = prepare_linear_weight(shared_down);
    const auto gate_format  = gate_up_bank.weight.qtype;
    const auto down_format  = down_bank.weight.qtype;
    require(router_bank.weight.qtype == QType::BF16 && shared.weight.qtype == QType::Q8_G32_FP16 &&
                down.weight.qtype == QType::Q8_G32_FP16 &&
                ((gate_format == QType::Q4_G64_FP16 &&
                  (down_format == QType::Q5_G64_FP16 || down_format == QType::Q6_G64_FP16)) ||
                 (gate_format == QType::Q8_G32_FP16 && down_format == QType::Q8_G32_FP16)),
            "SparseMoe native bank formats are unsupported");
    return {router_bank.weight, gate_up_bank.weight, down_bank.weight, shared.weight, down.weight};
}

} // namespace ninfer::ops
