// ninfer::ops - causal cached Softmax Attention validation and finite route dispatch.
#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim                      = 256;
constexpr float kExpectedScale                       = 0.0625f;
constexpr std::int32_t kMaximumVerifyTokens          = 16;
constexpr std::int32_t kMaximumBatchSize             = 8;
constexpr std::uint32_t kTwoChunkPromptVisibleKeys   = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

std::int32_t causal_attention_chunk_tokens(std::int32_t q_heads, std::int32_t width,
                                           std::int32_t batch_size, KvCacheStorage storage,
                                           CausalAttentionExecutionEnvelope envelope) {
    if (q_heads == 16) return 6;
    // Balance the two narrow BF16 chunks; INT8 benefits from 5+4/5 at long contexts.
    if (batch_size == 1 && ((storage == KvCacheStorage::BFloat16 && width >= 9 && width <= 12) ||
                            (storage == KvCacheStorage::Int8Group64 && width >= 9 && width <= 10 &&
                             envelope.max_visible_keys > 4096)))
        return (width + 1) / 2;
    return 8;
}

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        !((geometry.query_heads == 24 && geometry.kv_heads == 4) ||
          (geometry.query_heads == 16 && geometry.kv_heads == 2))) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        std::fprintf(stderr,
                     "[kv-dtype-diag] direct storage=%d expected k=%d v=%d actual k=%d v=%d "
                     "k_ne0=%d v_ne0=%d\n",
                     static_cast<int>(cache.storage), static_cast<int>(layout.key.data_dtype),
                     static_cast<int>(layout.value.data_dtype),
                     static_cast<int>(cache.k_pages.dtype),
                     static_cast<int>(cache.v_pages.dtype), cache.k_pages.ne[0],
                     cache.v_pages.ne[0]);
        std::fflush(stderr);
        throw std::invalid_argument(std::string(op) + ": invalid KV cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, op, name);
        require_contiguous_nonnull(tensor, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t validate_batch_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_tables.ne[0];
    const std::int32_t table_rows     = cache.block_tables.ne[1];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        std::fprintf(stderr,
                     "[kv-dtype-diag] storage=%d expected k=%d v=%d actual k=%d v=%d "
                     "k_ne0=%d v_ne0=%d\n",
                     static_cast<int>(cache.storage), static_cast<int>(layout.key.data_dtype),
                     static_cast<int>(layout.value.data_dtype),
                     static_cast<int>(cache.k_pages.dtype),
                     static_cast<int>(cache.v_pages.dtype), cache.k_pages.ne[0],
                     cache.v_pages.ne[0]);
        std::fflush(stderr);
        throw std::invalid_argument(std::string(op) + ": invalid KV cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, op, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block tables must be I32");
    }
    require_shape(cache.block_tables, logical_pages, table_rows, 1, 1, op, "block tables");
    require_contiguous_nonnull(cache.block_tables, op, "block tables");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, op, name);
        require_contiguous_nonnull(tensor, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void validate_envelope(CausalAttentionExecutionEnvelope envelope, const PagedKVLayerView& cache,
                       std::int32_t tokens, const char* op) {
    const std::uint32_t capacity = validate_cache(cache, cache.num_kv_heads, op);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                AttentionHeadGeometry geometry, const PagedKVLayerView& cache,
                                CausalAttentionExecutionEnvelope envelope, float scale,
                                const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const PagedKVBatchLayerView& cache,
                                        AttentionHeadGeometry geometry,
                                        CausalAttentionExecutionEnvelope envelope, float scale,
                                        const char* op) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatchSize ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    require_shape(q, kHeadDim, q_heads, width, batch, op, "q");
    require_shape(positions, width, batch, 1, 1, op, "positions");
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns"); }
    require_shape(kv_table_rows, batch, 1, 1, 1, op, "KV table rows");
    require_shape(out, kHeadDim, q_heads, width, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(kv_table_rows, op, "KV table rows");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op);
    if (cache.block_tables.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        (!masked && envelope.max_visible_keys < static_cast<std::uint32_t>(width))) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                           std::int32_t tokens, std::int32_t splits,
                                           std::int32_t batch_size) {
    return {
        workspace.alloc(DType::FP32, {kHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

template <typename Launch>
void for_each_small_t_chunk(const Tensor& q, const Tensor& positions, WorkspaceArena& workspace,
                            KvCacheStorage cache_storage, CausalAttentionExecutionEnvelope envelope,
                            Tensor& out, Launch&& launch) {
    for (std::int32_t begin = 0; begin < q.ne[2];
         begin +=
         causal_attention_chunk_tokens(q.ne[1], q.ne[2], 1, cache_storage, envelope)) {
        const std::int32_t count = std::min(
            causal_attention_chunk_tokens(q.ne[1], q.ne[2], 1, cache_storage, envelope),
            q.ne[2] - begin);
        auto chunk_scope = workspace.scope();
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], count, cache_storage, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], count, splits, 1);
        Tensor q_chunk          = q.slice(2, begin, count);
        Tensor position_chunk   = positions.slice(0, begin, count);
        Tensor out_chunk        = out.slice(2, begin, count);
        launch(begin, count, q_chunk, position_chunk, partial, out_chunk);
    }
}

void launch_chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid_columns,
                            const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream) {
    for (std::int32_t begin = 0; begin < q.ne[2];
         begin += causal_attention_chunk_tokens(q.ne[1], q.ne[2], q.ne[3], cache.storage,
                                                        envelope)) {
        const std::int32_t count  = std::min(causal_attention_chunk_tokens(
                                                q.ne[1], q.ne[2], q.ne[3], cache.storage, envelope),
                                             q.ne[2] - begin);
        auto chunk_scope          = workspace.scope();
        const std::int32_t splits = detail::causal_attention_split_capacity(
            q.ne[1], count, cache.storage, envelope, q.ne[3]);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], count, splits, q.ne[3]);
        detail::causal_attention_small_t_launch(q, k, v, positions, valid_columns, table_rows,
                                                scale, cache, envelope, begin, count, partial.acc,
                                                partial.m, partial.l, out, stream);
    }
}

void launch_cached_chunked_small_t(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.storage, envelope, out,
        [&](std::int32_t, std::int32_t, const Tensor& q_chunk, const Tensor& position_chunk,
            SmallTWorkspace& partial, Tensor& out_chunk) {
            detail::causal_attention_cached_small_t_launch(q_chunk, position_chunk, scale, cache,
                                                           envelope, partial.acc, partial.m,
                                                           partial.l, out_chunk, stream);
        });
}

} // namespace

namespace detail {

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope) {
    // AMD port (gfx1201): A/B probe knob for the speculative-verify tile. Width 1 never
    // reaches this decision (decode_t1_wmma dispatches first), so width > 1 is exactly the
    // verify/MTP tile; forcing it onto the well-exercised prompt kernel isolates whether a
    // rejection-rate bug lives in the small-T tiled kernel or in the shared inputs.
    static const bool force_prompt_verify = [] {
        const char* e = std::getenv("NINFER_ATTN_FORCE_PROMPT");
        return e != nullptr && e[0] == '1';
    }();
    if (force_prompt_verify && width > 1 && width <= kMaximumVerifyTokens) {
        return CausalAttentionRoute::Prompt;
    }
    if (q_heads == 24 && width <= kMaximumVerifyTokens) {
        if (batch_size == 1) {
            std::uint32_t prompt_limit = 0;
            // AMD port (gfx1201): A/B switch for the int4 WMMA prompt kernel. Default ON
            // (O(n) prefill); NINFER_INT4_PROMPT_WMMA=0 restores the chunked small-T walk,
            // which is the only reference implementation for this storage on this port.
            static const bool int4_prompt_wmma = [] {
                const char* e = std::getenv("NINFER_INT4_PROMPT_WMMA");
                return e == nullptr || e[0] != '0';
            }();
            switch (storage) {
            case KvCacheStorage::BFloat16:
                // AMD port (gfx1201): decode T=1 goes through the zero-shuffle WMMA prompt
                // kernel whenever the gate is on (it reads the KV cache at full bandwidth);
                // the old chunked-small-T route underutilized the GPU by 100x at T=1.
                if (storage == KvCacheStorage::BFloat16 && attn_prompt_wmma_enabled() &&
                    width == 1) {
                    return CausalAttentionRoute::Prompt;
                }
                prompt_limit = width <= 4 ? 128 : width <= 8 ? 256 : 640;
                break;
            case KvCacheStorage::Fp16:
                // Fp16 paged-KV storage: no small-T reader exists for fp16 K, so every
                // prompt-shaped call takes the fp16 WMMA prompt/decode kernel.
                if (attn_prompt_wmma_enabled() && width == 1) {
                    return CausalAttentionRoute::Prompt;
                }
                prompt_limit = width <= 4 ? 128 : width <= 8 ? 256 : 640;
                break;
            case KvCacheStorage::Q4_0:
                // q4_0 paged-KV storage: same routing as Fp16 (no small-T reader).
                if (attn_prompt_wmma_enabled() && width == 1) {
                    return CausalAttentionRoute::Prompt;
                }
                prompt_limit = width <= 4 ? 128 : width <= 8 ? 256 : 640;
                break;
            case KvCacheStorage::Int8Group64:
                // AMD port (gfx1201): the KInt8 WMMA prompt kernel (57KB smem) carries
                // widths 9-16; width <= 8 stays on the int8 small-T decode path.
                prompt_limit = width <= 8 ? 0 : 256;
                break;
            case KvCacheStorage::Fp8E4M3Row256:
                prompt_limit = width <= 4 ? 0 : width <= 8 ? 128 : 320;
                break;
            case KvCacheStorage::Nvfp4Group16:
                prompt_limit = width <= 8 ? 0 : 256;
                break;
            case KvCacheStorage::Fp8KeyNvfp4Value:
                prompt_limit = width <= 4 ? 0 : width <= 8 ? 128 : 320;
                break;
            case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
                // AMD port (gfx1201): prompt.cu now carries a RotatedInt4KeyInt4ValueE8
                // branch (packed-nibble K/V decode, rotated-Q int8 staging, bf16 mma), so
                // widths above 8 take the WMMA prompt kernel like Int8Group64 does. The
                // int4 small-T reader stays authoritative for width <= 8 decode tiles.
                prompt_limit = (!int4_prompt_wmma || width <= 8) ? 0 : 640;
                break;
            }
            // AMD port (gfx1201): the upstream fused prompt kernels request 85-98KB dynamic
            // smem and cannot launch here. The zero-shuffle WMMA prompt kernel
            // (prompt_bf16_wmma.cuh) needs only 40KB and carries the prompt route instead;
            // roll back with NINFER_ATTN_PROMPT_WMMA=0 (then every prompt-shaped call goes
            // through the chunked small-T path, <=8-token tiles).
            if (envelope.max_visible_keys <= prompt_limit) return CausalAttentionRoute::Prompt;
        }
        return width <= 8 ? CausalAttentionRoute::SmallT : CausalAttentionRoute::Prompt;
    }
    if (width <= 6) {
        // AMD port (gfx1201): Fp16/Q4_0 paged-KV has no small-T bf16 reader; prefill-shaped
        // calls route through the fp16 WMMA prompt kernel instead.
        if (storage == KvCacheStorage::Fp16 || storage == KvCacheStorage::Q4_0) {
            return CausalAttentionRoute::Prompt;
        }
        return CausalAttentionRoute::SmallT;
    }
    if (batch_size > 1) return CausalAttentionRoute::ChunkedSmallT;
    const std::uint32_t prompt_visible_keys =
        width <= 12 ? kTwoChunkPromptVisibleKeys : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && width <= kMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys)
        return CausalAttentionRoute::ChunkedSmallT;
    // AMD port (gfx1201): prefill-sized prompt calls carry the zero-shuffle WMMA
    // kernel (40KB smem, launches on gfx12) — BF16 / FP16 / Q4_0 KV storage, and
    // Int8Group64 through its KInt8 twin (57KB smem, rotated-Q staging; the
    // upstream Int8 prompt kernel requests 85-98KB smem and cannot launch on gfx12,
    // so Int8Group64 used to fall back to chunked small-T — 85 min @100k). Roll
    // back with NINFER_ATTN_PROMPT_WMMA=0.
    // RotatedInt4KeyInt4ValueE8 joins this list: prompt.cu now carries an int4 twin of
    // the WMMA kernel (packed-nibble K/V decode + rotated-Q int8 staging, bf16 mma), so
    // prefill is O(n) instead of the chunked small-T O(n^2) walk (measured 149 t/s at
    // 5.6k tokens and ~80 min extrapolated at 64k before this change).
    // int4 is gated by the same switch used above so the chunked small-T reference stays
    // reachable for A/B validation (NINFER_INT4_PROMPT_WMMA=0).
    static const bool int4_prompt_wmma_list = [] {
        const char* e = std::getenv("NINFER_INT4_PROMPT_WMMA");
        return e == nullptr || e[0] != '0';
    }();
    if (storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Fp16 ||
        storage == KvCacheStorage::Q4_0 || storage == KvCacheStorage::Int8Group64 ||
        (storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 && int4_prompt_wmma_list)) {
        return CausalAttentionRoute::Prompt;
    }
    return CausalAttentionRoute::ChunkedSmallT;
}

const char* causal_attention_route_name(CausalAttentionRoute route) {
    switch (route) {
    case CausalAttentionRoute::SmallT:
        return "small_t";
    case CausalAttentionRoute::ChunkedSmallT:
        return "chunked_small_t";
    case CausalAttentionRoute::Prompt:
        return "prompt";
    }
    return "unknown";
}

} // namespace detail

std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
    CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width) {
    require_causal_geometry(geometry, "causal_softmax_attention workspace");
    const std::int32_t q_heads = geometry.query_heads;
    bool supported_dtype       = true;
    try {
        (void)paged_kv_storage_layout(cache_storage, kHeadDim);
    } catch (const std::invalid_argument&) { supported_dtype = false; }
    if (!supported_dtype || batch_size <= 0 || batch_size > kMaximumBatchSize || min_width <= 0 ||
        max_width < min_width || (batch_size > 1 && max_width > kMaximumVerifyTokens) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument(
            "causal_softmax_attention workspace: invalid profile or interval");
    }

    const auto chunk_capacity = [&](std::int32_t width) {
        const std::int32_t splits = detail::causal_attention_split_capacity(
            q_heads, width, cache_storage, envelope, batch_size);
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, q_heads, width, splits, batch_size);
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t width) {
        const detail::CausalAttentionRoute route = detail::causal_attention_resolve_route(
            q_heads, width, batch_size, cache_storage, envelope);
        if (route == detail::CausalAttentionRoute::Prompt) { return std::size_t{0}; }
        if (route == detail::CausalAttentionRoute::SmallT) { return chunk_capacity(width); }
        std::size_t maximum = 0;
        for (std::int32_t begin = 0; begin < width;
             begin += causal_attention_chunk_tokens(q_heads, width, batch_size,
                                                            cache_storage, envelope)) {
            maximum = std::max(
                maximum,
                chunk_capacity(std::min(causal_attention_chunk_tokens(
                                            q_heads, width, batch_size, cache_storage, envelope),
                                        width - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    if (min_width <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_width, kMaximumVerifyTokens);
        for (std::int32_t width = min_width; width <= last; ++width) {
            maximum = std::max(maximum, exact_capacity(width));
        }
    }
    return maximum;
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "causal_softmax_attention";
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       geometry, envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("causal_softmax_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = geometry.kv_heads;
    require_shape(k, kHeadDim, kv_heads, width, batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    auto scope = workspace.scope();
    const detail::CausalAttentionRoute route =
        detail::causal_attention_resolve_route(q.ne[1], width, batch, cache.storage, envelope);
    {
        // AMD port: one-shot route trace (NINFER_ATTN_TRACE=1 keeps every call).
        static const bool trace_all = [] {
            const char* e = std::getenv("NINFER_ATTN_TRACE");
            return e != nullptr && e[0] == '1';
        }();
        static bool route_printed = false;
        if (trace_all || !route_printed) {
            route_printed = true;
            std::fprintf(stderr, "[attn-route] q_heads=%d width=%d batch=%d route=%s\n", q.ne[1],
                         width, batch, detail::causal_attention_route_name(route));
            std::fflush(stderr);
        }
    }
    if (route == detail::CausalAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, valid_columns, kv_table_rows, scale, cache,
                               envelope, workspace, out, stream);
        return;
    }
    // AMD port (gfx1201) DEBUG: exact A/B between the T=1 WMMA kernel and the SmallT
    // reference, on the SAME staged cache. NINFER_ATTN_XCHECK=1 (requires --no-graph:
    // the host sync below is illegal inside a stream capture).
    static const bool xcheck_on = [] {
        const char* e = std::getenv("NINFER_ATTN_XCHECK");
        return e != nullptr && e[0] == '1';
    }();
    if (xcheck_on && width == 1 && batch == 1 && cache.storage == KvCacheStorage::BFloat16 &&
        q.ne[1] == 24) {
        detail::causal_decode_t1_wmma_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                             scale, cache, envelope, workspace, out, stream);
        Tensor ref = workspace.alloc(DType::BF16, {out.ne[0], out.ne[1], out.ne[2], out.ne[3]});
        const std::int32_t xsplits = detail::causal_attention_split_capacity(
            q.ne[1], width, cache.storage, envelope, batch);
        SmallTWorkspace xp = allocate_small_t_workspace(workspace, q.ne[1], width, xsplits, batch);
        detail::causal_attention_small_t_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                                scale, cache, envelope, 0, width, xp.acc, xp.m,
                                                xp.l, ref, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t nelt = static_cast<std::size_t>(out.ne[0]) * out.ne[1] * out.ne[2] *
                                 out.ne[3];
        std::vector<std::uint16_t> va(nelt), vb(nelt);
        CUDA_CHECK(cudaMemcpy(va.data(), out.data, nelt * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vb.data(), ref.data, nelt * 2, cudaMemcpyDeviceToHost));
        static int xc_calls = 0;
        ++xc_calls;
        if (xc_calls <= 48) {
            const auto b2f = [](std::uint16_t h) {
                std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
                float f;
                std::memcpy(&f, &bits, 4);
                return f;
            };
            float worst = 0.0f;
            int worst_h = -1, worst_d = -1, bad_heads = 0;
            for (std::int32_t h = 0; h < q.ne[1]; ++h) {
                float wh = 0.0f, wv = 0.0f;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t idx = static_cast<std::size_t>(h) * kHeadDim + d;
                    const float av = b2f(va[idx]), bv = b2f(vb[idx]);
                    const float diff = std::fabs(av - bv);
                    if (diff > wh) { wh = diff; }
                    if (std::fabs(bv) > wv) { wv = std::fabs(bv); }
                    if (diff > worst) { worst = diff; worst_h = h; worst_d = d; }
                }
                if (wv > 0.0f && wh / wv > 0.02f) { ++bad_heads; }
                if (h < 6) {
                    std::fprintf(stderr, "[xchk] call=%d head=%d maxdiff=%g |ref|max=%g%c", xc_calls,
                                 h, wh, wv, 10);
                }
            }
            std::fprintf(stderr, "[xchk] call=%d worst=%g head=%d d=%d bad_heads=%d%c", xc_calls,
                         worst, worst_h, worst_d, bad_heads, 10);
            std::fflush(stderr);
        }
        return;
    }
    // AMD port (gfx1201): T=1 decode (main path, fresh K/V staged into the cache) takes
    // the zero-shuffle WMMA split-KV kernel. Gated by NINFER_DECODE_WMMA (default ON
    // since the prefill-WMMA fallback NaN'd at @100k; roll back with =0). BFloat16 /
    // Fp16 / Q4_0 / Int8Group64 / Fp8 / rk4v4-e8 all route here.
    if (width == 1 && batch == 1 &&
        (cache.storage == KvCacheStorage::BFloat16 || cache.storage == KvCacheStorage::Fp16 ||
         cache.storage == KvCacheStorage::Q4_0 ||
         cache.storage == KvCacheStorage::Fp8E4M3Row256 ||
         cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
         cache.storage == KvCacheStorage::Int8Group64) &&
        detail::decode_t1_wmma_enabled()) {
        detail::causal_decode_t1_wmma_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                             scale, cache, envelope, workspace, out, stream);
        return;
    }
    // AMD port (gfx1201): 2..8 tokens from ONE sequence (the speculative-verify batch)
    // ride the tuned T=1 SIMT kernel once per token. Measured at a 5.6k window: the
    // small-T route spends 78.2 ms of a 119.2 ms verify round in attention (~270x the
    // bandwidth floor), the per-token SIMT route ~0.1 ms/layer per token.
    // NINFER_VERIFY_SIMT=0 restores the small-T route for A/B measurement.
    static const bool verify_simt_on = [] {
        const char* e = std::getenv("NINFER_VERIFY_SIMT");
        return e == nullptr || e[0] != '0';
    }();
    if (verify_simt_on && batch == 1 && width > 1 && width <= 8) {
        const bool took = detail::causal_decode_t1_simt_multi_token_launch(
            q, k, v, positions, kv_table_rows, scale, cache, envelope, width, workspace, out,
            stream);
        static int mt_route_calls = 0;
        if (std::getenv("NINFER_MTDBG") != nullptr && mt_route_calls++ < 400) {
            std::int32_t pos0 = -1, posn = -1;
            std::uint16_t out0 = 0;
            if (positions.data != nullptr) {
                CUDA_CHECK(cudaMemcpy(&pos0, positions.data, sizeof(pos0),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&posn,
                                      static_cast<const std::int32_t*>(positions.data) + (width - 1),
                                      sizeof(posn), cudaMemcpyDeviceToHost));
            }
            std::int32_t trow0 = -1;
            if (kv_table_rows.data != nullptr) {
                CUDA_CHECK(cudaMemcpy(&trow0, kv_table_rows.data, sizeof(trow0),
                                      cudaMemcpyDeviceToHost));
            }
            if (out.data != nullptr) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaMemcpy(&out0, out.data, sizeof(out0), cudaMemcpyDeviceToHost));
            }
            std::fprintf(stderr,
                         "[mt-route] width=%d storage=%d took=%d env=[%u,%u] pos=%d..%d trow=%d "
                         "out0=%04x kpages=%p rows=%p out=%p\n",
                         width, static_cast<int>(cache.storage), took ? 1 : 0,
                         static_cast<unsigned>(envelope.min_visible_keys),
                         static_cast<unsigned>(envelope.max_visible_keys), pos0, posn, trow0,
                         static_cast<unsigned>(out0), cache.k_pages.data, kv_table_rows.data,
                         out.data);
            std::fflush(stderr);
        }
        if (took) { return; }
    }
    if (route == detail::CausalAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], width, cache.storage, envelope, batch);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], width, splits, batch);
        detail::causal_attention_small_t_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                                scale, cache, envelope, 0, width, partial.acc,
                                                partial.m, partial.l, out, stream);
        return;
    }
    {
        // AMD port: route trace, debug only (was unconditional and interleaved with chat output).
        static const bool attn_trace_on = std::getenv("NINFER_ATTN_TRACE") != nullptr;
        static bool fb_printed = false;
        if (attn_trace_on && !fb_printed) {
            fb_printed = true;
            std::fprintf(stderr, "[attn-fallback] prompt_launch width=%d route=%s maxvis=%u%c", width,
                         detail::causal_attention_route_name(route),
                         static_cast<unsigned>(envelope.max_visible_keys), 10);
            std::fflush(stderr);
        }
    }
    detail::causal_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                           cache, out, stream);
}

void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const PagedKVLayerView& cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "causal_softmax_attention_cached";
    validate_attention_tensors(q, positions, out, geometry, cache, envelope, scale, op);

    auto scope = workspace.scope();
    const detail::CausalAttentionRoute route =
        detail::causal_attention_resolve_route(q.ne[1], q.ne[2], 1, cache.storage, envelope);
    if (route == detail::CausalAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out, stream);
        return;
    }
    // AMD port (gfx1201): T=1 decode takes the dedicated zero-shuffle WMMA split-KV
    // kernel (decode_t1_wmma.cuh) + the production small-T reducer. Gated by
    // NINFER_DECODE_WMMA (default ON; =0 rolls back). BFloat16/Fp16/Q4_0/Int8Group64/
    // Fp8/rk4v4-e8 all route here.
    // AMD port (gfx1201): the cached T=1 WMMA variant is only reached by the MTP head
    // (models/qwen3_5/execution/text.cpp mtp_prefill_chunk -> causal_softmax_attention_cached);
    // the main decode goes through the *append* variant above. Measured 2026-10-28: the cached
    // WMMA kernel writes NaN/-inf partials (m stays at the -inf sentinel, l carries NaN) for
    // BFloat16 AND rk4v4-e8, i.e. every key is dropped and the attention branch contributes
    // exactly zero to the MTP logits. Its partials come back healthy through the tuned SIMT
    // small-T kernel (||a||=164 vs 0.000), so the cached route defaults to SIMT until the WMMA
    // split derivation is fixed. NINFER_CACHED_WMMA=1 re-enables the old path for A/B.
    static const bool cached_wmma_on = [] {
        const char* e = std::getenv("NINFER_CACHED_WMMA");
        return e != nullptr && e[0] == '1';
    }();
    if (cached_wmma_on && q.ne[2] == 1 &&
        (cache.storage == KvCacheStorage::BFloat16 || cache.storage == KvCacheStorage::Fp16 ||
         cache.storage == KvCacheStorage::Q4_0 ||
         cache.storage == KvCacheStorage::Fp8E4M3Row256 ||
         cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
         cache.storage == KvCacheStorage::Int8Group64) &&
        detail::decode_t1_wmma_enabled()) {
        detail::causal_decode_t1_wmma_cached_launch(q, positions, scale, cache, envelope, workspace,
                                                    out, stream);
        return;
    }
    if (route == detail::CausalAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::causal_attention_split_capacity(q.ne[1], q.ne[2], cache.storage, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], q.ne[2], splits, 1);
        detail::causal_attention_cached_small_t_launch(
            q, positions, scale, cache, envelope, partial.acc, partial.m, partial.l, out, stream);
        return;
    }
    detail::causal_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops




