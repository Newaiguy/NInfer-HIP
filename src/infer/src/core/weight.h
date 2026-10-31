#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // Prism-private ternary (Bonsai 2 27B): group 128, base-3 / 2-bit codes.
    // AMD port: ternary support from shensanshu/ninfer-ada-ternary.
    PTQ1_0_G128         = 9,
    PQ2_0_G128          = 10,
    // iamwavecut/ninfer-all Bonsai v3: same row-split G128 ternary layout, but the
    // converter maps GGUF PQ2_0 codes into a (bit0 select / bit1 sign) encoding before
    // storing, so loading a v3 artifact must use the t2 codec, not the raw PQ2 mapping.
    T2_G128_FP16        = 11,
    // ggml block types kept as a GGUF stored them (GgufBlocks layout only): each row is its K /
    // block_elements blocks, byte for byte. Their products quantize the activation to ggml's q8_1.
    // Values are offset to 12 so the ternary codes above keep their numeric ids.
    GGUF_Q8_0    = 12,
    GGUF_Q2_K    = 13,
    GGUF_Q3_K    = 14,
    GGUF_Q4_K    = 15,
    GGUF_Q5_K    = 16,
    GGUF_Q6_K    = 17,
    GGUF_IQ2_XXS = 18,
    GGUF_IQ2_XS  = 19,
    GGUF_IQ2_S   = 20,
    GGUF_IQ3_XXS = 21,
    GGUF_IQ3_S   = 22,
    GGUF_IQ1_S   = 23,
    GGUF_IQ1_M   = 24,
    GGUF_IQ4_NL  = 25,
    GGUF_IQ4_XS  = 26,
};

[[nodiscard]] constexpr bool is_gguf(QType format) {
    return format >= QType::GGUF_Q8_0 && format <= QType::GGUF_IQ4_XS;
}

struct GgufBlockShape {
    int elements = 0;
    int bytes    = 0;
};

// ggml-common.h's block sizes; {0, 0} for every non-GGUF format.
[[nodiscard]] constexpr GgufBlockShape gguf_block_shape(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return {32, 34};
    case QType::GGUF_Q2_K: return {256, 84};
    case QType::GGUF_Q3_K: return {256, 110};
    case QType::GGUF_Q4_K: return {256, 144};
    case QType::GGUF_Q5_K: return {256, 176};
    case QType::GGUF_Q6_K: return {256, 210};
    case QType::GGUF_IQ2_XXS: return {256, 66};
    case QType::GGUF_IQ2_XS: return {256, 74};
    case QType::GGUF_IQ2_S: return {256, 82};
    case QType::GGUF_IQ3_XXS: return {256, 98};
    case QType::GGUF_IQ3_S: return {256, 110};
    case QType::GGUF_IQ1_S: return {256, 50};
    case QType::GGUF_IQ1_M: return {256, 56};
    case QType::GGUF_IQ4_NL: return {32, 18};
    case QType::GGUF_IQ4_XS: return {256, 136};
    default: return {};
    }
}

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // Rows of whole ggml blocks, as a GGUF stores them.
    GgufBlocks = 5,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // Optional INT32 [K] gather of the input columns for a GGUF matrix whose stored columns
    // are a permutation of its logical input's: column c multiplies input element
    // input_columns[c]. Set only on GDN output projections (see gguf_blocks tiled value heads);
    // null for every other matrix.
    const std::int32_t* input_columns = nullptr;

    // Prism ternary weights are stored folded into a rotated basis: the model computes
    // y = W' * (H * (s * P * x)), so every activation feeding one of the folded weights must
    // be mapped by (signs, then normalized Sylvester-Hadamard) before the matmul, and the
    // token-embedding lookup must be mapped by (rotate, then signs) afterwards.
    //
    // hadamard_signs points at the sign block for THIS weight's input width (each width owns
    // k/1024 rows of 1024 F32 +-1 values, and row b within the width is selected by the
    // activation's 1024-block index). nullptr means no transform. hadamard_n_blk = k/1024.
    const float* hadamard_signs = nullptr;
    std::int32_t hadamard_n_blk = 0;

    // Folded-basis feature permutation P, applied BEFORE the signs and the rotation:
    //
    //     x.view(perm_hd, perm_nk, perm_rep) -> swap the last two axes -> flatten
    //
    // Only the GDN output projection stores grouped columns while the runtime produces tiled
    // v-heads, so perm_rep > 1 enables it and everything else leaves it at 1. Geometry comes
    // from the model's head counts: perm_hd = K/n_v, perm_nk = n_k, perm_rep = n_v/n_k.
    std::int32_t hadamard_perm_hd  = 0;
    std::int32_t hadamard_perm_nk  = 0;
    std::int32_t hadamard_perm_rep = 1;
};

} // namespace ninfer
