#include "artifact/formats.h"

#include "artifact/schema.h"

#include <array>

namespace ninfer::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{QType::BF16, std::string_view{"bf16"}},
    std::pair{QType::FP32, std::string_view{"fp32"}},
    std::pair{QType::INT32, std::string_view{"int32"}},
    std::pair{QType::Q4_G64_FP16, std::string_view{"q4_g64_fp16"}},
    std::pair{QType::Q5_G64_FP16, std::string_view{"q5_g64_fp16"}},
    std::pair{QType::Q6_G64_FP16, std::string_view{"q6_g64_fp16"}},
    std::pair{QType::Q8_G32_FP16, std::string_view{"q8_g32_fp16"}},
    std::pair{QType::NVFP4, std::string_view{"nvfp4"}},
    std::pair{QType::FP8_E4M3FN_ROW_BF16, std::string_view{"fp8_e4m3fn_row_bf16"}},
    // AMD port: ternary formats from shensanshu/ninfer-ada-ternary.
    std::pair{QType::PTQ1_0_G128, std::string_view{"ptq1_0_g128"}},
    std::pair{QType::PQ2_0_G128, std::string_view{"pq2_0_g128"}},
    // iamwavecut/ninfer-all Bonsai v3 ternary (bit0 select / bit1 sign codec).
    std::pair{QType::T2_G128_FP16, std::string_view{"t2_g128_fp16"}},
    // ggml block types kept byte-for-byte as a GGUF stored them (GgufBlocks layout).
    std::pair{QType::GGUF_Q8_0, std::string_view{"gguf_q8_0"}},
    std::pair{QType::GGUF_Q2_K, std::string_view{"gguf_q2_k"}},
    std::pair{QType::GGUF_Q3_K, std::string_view{"gguf_q3_k"}},
    std::pair{QType::GGUF_Q4_K, std::string_view{"gguf_q4_k"}},
    std::pair{QType::GGUF_Q5_K, std::string_view{"gguf_q5_k"}},
    std::pair{QType::GGUF_Q6_K, std::string_view{"gguf_q6_k"}},
    std::pair{QType::GGUF_IQ2_XXS, std::string_view{"gguf_iq2_xxs"}},
    std::pair{QType::GGUF_IQ2_XS, std::string_view{"gguf_iq2_xs"}},
    std::pair{QType::GGUF_IQ2_S, std::string_view{"gguf_iq2_s"}},
    std::pair{QType::GGUF_IQ3_XXS, std::string_view{"gguf_iq3_xxs"}},
    std::pair{QType::GGUF_IQ3_S, std::string_view{"gguf_iq3_s"}},
    std::pair{QType::GGUF_IQ1_S, std::string_view{"gguf_iq1_s"}},
    std::pair{QType::GGUF_IQ1_M, std::string_view{"gguf_iq1_m"}},
    std::pair{QType::GGUF_IQ4_NL, std::string_view{"gguf_iq4_nl"}},
    std::pair{QType::GGUF_IQ4_XS, std::string_view{"gguf_iq4_xs"}},
};
constexpr std::array kLayouts = {
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous_le_v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row_split_k128_v1"}},
    std::pair{QuantLayout::RowScale, std::string_view{"row_scale_v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4, std::string_view{"block_scale_k16_m128x4_v1"}},
    std::pair{QuantLayout::GgufBlocks, std::string_view{"gguf_blocks_v1"}},
};

} // namespace

QType parse_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    // AMD port: shensanshu pack.py spells formats in the v1.0.8 uppercase convention.
    if (name == "PTQ1_0_G128") { return QType::PTQ1_0_G128; }
    if (name == "PQ2_0_G128") { return QType::PQ2_0_G128; }
    if (name == "BF16") { return QType::BF16; }
    if (name == "FP32") { return QType::FP32; }
    if (name == "I32") { return QType::INT32; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    // AMD port: pack.py spells layouts with dashes (v1.0.8 convention).
    if (name == "row-split-k128-v1") { return QuantLayout::RowSplit; }
    if (name == "contiguous-le-v1") { return QuantLayout::Contiguous; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept {
    for (const auto& [value, spelling] : kFormats) {
        if (value == format) { return spelling; }
    }
    return {};
}

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninfer::artifact
