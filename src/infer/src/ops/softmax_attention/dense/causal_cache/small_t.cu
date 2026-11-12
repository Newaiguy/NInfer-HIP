// ninfer::ops - split-KV causal small-T launcher and unified route dispatcher. INT8 Q/K
// preparation, including their paired fixed rotation, remains private to the included kernel.
#include "ops/softmax_attention/dense/causal_cache/key_position_hook.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail {
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t causal_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::SmallTMaximumSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::SmallTSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::SmallTSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::SmallTSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::SmallTSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::SmallTSplitScale); }

    return (splits < Geometry::SmallTMaximumSplits) ? splits : Geometry::SmallTMaximumSplits;
}

template <typename Geometry>
std::int32_t causal_small_t_split_count(std::int32_t window, std::int32_t tokens,
                                        KvCacheStorage storage) {
    if constexpr (Geometry::SmallTSplitScale == 1) {
        if (storage == KvCacheStorage::Fp8E4M3Row256 && tokens == 1 && window > 8198) {
            return Geometry::SmallTMaximumSplits;
        }
    }
    // A 64-key default split just above a 32-key boundary makes the partial kernel execute a
    // nearly empty second tile. T=5 uses one 32-key tile per split; the short T>=6 profile keeps
    // all newly appended rows in one tail split while retaining a useful B=8 grid.
    if (storage == KvCacheStorage::Int8Group64 && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::SmallTSplitScale);
    }
    if (storage == KvCacheStorage::Int8Group64 && tokens >= 6 && window > 128 && window <= 160) {
        constexpr std::int32_t kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
        return div_up(window, kKeysPerSplit);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (storage == KvCacheStorage::Int8Group64 && tokens >= 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::SmallTSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::SmallTSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::SmallTSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return causal_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t causal_small_t_launch_capacity(CausalAttentionExecutionEnvelope envelope,
                                            std::int32_t tokens, KvCacheStorage storage) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits = causal_small_t_split_count<Geometry>(static_cast<std::int32_t>(window),
                                                                    tokens, storage);
        capacity          = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    // gfx1201 (32 CU) port: the stock tiers are sized for a 170-SM part, so on this card the
    // decode grid comes out at ~2 CTAs/SM and the tiled partial kernels are latency-starved.
    // Raise the capacity floor to the split count that actually fills the device we are on
    // (4 CTAs per SM, one per kv head). The allocation below grows with it, so the device-side
    // active-split policy stays clamped to a real buffer.
    static const std::int32_t device_cu = [] {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) { return 0; }
        return static_cast<std::int32_t>(props.multiProcessorCount);
    }();
    if (device_cu > 0) {
        // Env override (NINFER_SPLIT_CTAS_PER_SM, default 4) so the decode split count can be
        // swept at long context without a rebuild: the tiled partial kernels are latency-starved
        // when 32 splits each have to walk 3000 keys at 96k ctx.
        static const std::int32_t ctas_per_sm = [] {
            const char* e = std::getenv("NINFER_SPLIT_CTAS_PER_SM");
            const int v    = e != nullptr ? std::atoi(e) : 0;
            return v > 0 ? v : 4;
        }();
        const std::int32_t kCtasPerSm = ctas_per_sm;
        const std::int32_t target = (kCtasPerSm * device_cu + Geometry::KVHeads - 1) / Geometry::KVHeads;
        if (capacity < target) { capacity = target; }
        if (capacity > Geometry::SmallTMaximumSplits) { capacity = Geometry::SmallTMaximumSplits; }
    }
    return capacity;
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging).
    causal_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                    Masked, CacheInput>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
            static_cast<__half*>(cache_v.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.width, invocation.full_width,
            invocation.column_begin, logical_capacity, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool PackedValues = false, KvKeyCoding Keys = KvKeyCoding::Int8>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                         MinBlocksPerSm, KeyBlock, DynamicArena,
                                                         MultiBatch, Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                                 KeyBlock, DynamicArena, MultiBatch, Masked,
                                                 CacheInput, PackedValues, Keys>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                invocation.key_position_map, invocation.resident_slots);
    };
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            // gfx12 port: the Bc=64 dynamic arena requests 4*64*256 = 64 KiB dynamic
            // smem, which already exceeds the 64 KiB per-block cap before the kernel's
            // static smem is counted, so cudaFuncSetAttribute fails with
            // hipErrorInvalidValue and every (2054, 8198] window aborts. Route those
            // windows through the 32-key non-dynamic configuration instead.
            launch.template operator()<12, 1, 32, false>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size) {
    if (tokens < 1 || tokens > (q_heads == 24 ? 8 : 6) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("causal_softmax_attention split capacity: invalid profile");
    }
    (void)paged_kv_storage_layout(cache_storage, kCausalHeadDim);
    if (q_heads == CausalD256H24Kv4::QHeads) {
        const int capacity =
            causal_small_t_launch_capacity<CausalD256H24Kv4>(envelope, tokens, cache_storage);
        if (batch_size > 1) {
            // Keep complete grids within one or two 170-SM waves. Rounding from 160 CTAs
            // leaves room for the indivisible 4*B group, including B=3/5/6/7.
            const bool narrow = tokens <= 5;
            int target_ctas   = 160;
            if (cache_storage == KvCacheStorage::BFloat16)
                target_ctas =
                    narrow || batch_size >= 5 || envelope.max_visible_keys > 4096 ? 320 : 160;
            else if (cache_storage == KvCacheStorage::Int8Group64)
                target_ctas = narrow || envelope.max_visible_keys > 4096 ? 320 : 160;
            else if (cache_storage == KvCacheStorage::Nvfp4Group16)
                target_ctas = narrow ? 320 : 160;
            const int grid_limit = div_up(target_ctas, 4 * batch_size);
            // A split stages at most 64 physical-page IDs. Leave two 64-key pages for
            // key-tile rounding and page alignment at the 262144-key resource limit.
            const int page_limit = div_up(static_cast<int>(envelope.max_visible_keys), 3968);
            return std::min(capacity, std::max({4, grid_limit, page_limit}));
        }
        return capacity;
    }
    if (q_heads == CausalD256H16Kv2::QHeads) {
        return causal_small_t_launch_capacity<CausalD256H16Kv2>(envelope, tokens, cache_storage);
    }
    throw std::invalid_argument(
        "causal_softmax_attention split capacity: unsupported head geometry");
}

template <typename Geometry, typename CacheInput>
void causal_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                         float scale, PagedKVBatchLayerView cache,
                                         const CausalSmallTInvocation& invocation,
                                         CausalAttentionExecutionEnvelope envelope,
                                         Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                         Tensor& out, cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits                = causal_attention_split_capacity(
        Geometry::QHeads, invocation.width, cache.storage, envelope, invocation.batch_size);

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define NINFER_CAUSAL_SMALL_T_DISPATCH(TOKENS, WARPS)                                              \
    do {                                                                                           \
        const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {                          \
            if (cache.storage == KvCacheStorage::Int8Group64) {                                    \
                launch_tc_partial_i8<Geometry, (TOKENS), MultiBatch, Masked>(                      \
                    q, input, pos, scale, cache, invocation, logical_capacity,                     \
                    implementation_window, splits, partial_acc, partial_m, partial_l, stream);     \
            } else if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {               \
                launch_tc_partial_i8<Geometry, (TOKENS), MultiBatch, Masked,                       \
                                     std::remove_cvref_t<decltype(input)>, true,                  \
                                     KvKeyCoding::Int4E8>(                                        \
                    q, input, pos, scale, cache, invocation, logical_capacity,                     \
                    implementation_window, splits, partial_acc, partial_m, partial_l, stream);     \
            } else {                                                                               \
                launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(           \
                    q, input, pos, scale, cache, invocation, logical_capacity, splits,             \
                    partial_acc, partial_m, partial_l, stream);                                    \
            }                                                                                      \
        };                                                                                         \
        const bool masked = invocation.valid_columns != nullptr;                                   \
        if (invocation.batch_size == 1) {                                                          \
            if (masked) {                                                                          \
                launch_profile.template operator()<false, true>();                                 \
            } else {                                                                               \
                launch_profile.template operator()<false, false>();                                \
            }                                                                                      \
        } else if (masked) {                                                                       \
            launch_profile.template operator()<true, true>();                                      \
        } else {                                                                                   \
            launch_profile.template operator()<true, false>();                                     \
        }                                                                                          \
    } while (0)

    switch (invocation.width) {
    case 1:
        NINFER_CAUSAL_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        NINFER_CAUSAL_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        NINFER_CAUSAL_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        NINFER_CAUSAL_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        NINFER_CAUSAL_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        NINFER_CAUSAL_SMALL_T_DISPATCH(6, 4);
        break;
    case 7:
        if constexpr (Geometry::QHeads == 24) {
            NINFER_CAUSAL_SMALL_T_DISPATCH(7, 4);
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    case 8:
        if constexpr (Geometry::QHeads == 24) {
            NINFER_CAUSAL_SMALL_T_DISPATCH(8, 4);
            break;
        }
        throw std::invalid_argument("unsupported query-row tile");
    default:
        throw std::invalid_argument("causal_attention_small_t_launch: unsupported T");
    }
#undef NINFER_CAUSAL_SMALL_T_DISPATCH


    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = Geometry::QHeads == 24 ? 256 : 64;
    const auto launch_reduce   = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
        const dim3 grid(Geometry::QHeads, div_up(kCausalHeadDim, kDChunk),
                          invocation.width * invocation.batch_size);
        causal_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                        Offset><<<grid, kReduceBlock, 0, stream>>>(
            static_cast<const float*>(partial_acc.data), static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data), static_cast<const std::int32_t*>(pos.data),
            invocation.valid_columns
                  ? static_cast<const std::int32_t*>(invocation.valid_columns->data)
                  : nullptr,
            invocation.width, invocation.full_width, invocation.column_begin, invocation.batch_size,
            splits, static_cast<__nv_bfloat16*>(out.data));
    };
    const auto launch_profile = [&]<bool Int8, bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0)
            launch_reduce.template operator()<Int8, MultiBatch, Masked, false>();
        else
            launch_reduce.template operator()<Int8, MultiBatch, Masked, true>();
    };
    const auto launch_for_storage = [&]<bool Int8>() {
        if (invocation.batch_size == 1) {
            if (invocation.valid_columns)
                launch_profile.template operator()<Int8, false, true>();
            else
                launch_profile.template operator()<Int8, false, false>();
        } else if (invocation.valid_columns)
            launch_profile.template operator()<Int8, true, true>();
        else
            launch_profile.template operator()<Int8, true, false>();
    };
    if (cache.storage == KvCacheStorage::Int8Group64 ||
        cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8)
        launch_for_storage.template operator()<true>();
    else
        launch_for_storage.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}

void causal_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_small_t_k8v4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                             envelope, column_begin, width, partial_acc, partial_m,
                                             partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_small_t_fp8_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                            envelope, column_begin, width, partial_acc, partial_m,
                                            partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_small_t_nvfp4_launch(q, k, v, pos, valid_columns, table_rows, scale, cache,
                                              envelope, column_begin, width, partial_acc, partial_m,
                                              partial_l, out, stream);
        return;
    }
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    const CausalSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size       = q.ne[3],
        .key_position_map = g_attention_key_position_map,
        .resident_slots   = g_attention_key_slots,
    };
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_launch_for<CausalD256H24Kv4>(q, input, pos, scale, cache,
                                                              invocation, envelope, partial_acc,
                                                              partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_launch_for<CausalD256H16Kv2>(q, input, pos, scale, cache, invocation,
                                                          envelope, partial_acc, partial_m,
                                                          partial_l, out, stream);
}

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_cached_small_t_k8v4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                    partial_m, partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_cached_small_t_fp8_launch(q, pos, scale, cache, envelope, partial_acc,
                                                   partial_m, partial_l, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_cached_small_t_nvfp4_launch(q, pos, scale, cache, envelope, partial_acc,
                                                     partial_m, partial_l, out, stream);
        return;
    }
    const CausalCachedInput input{};
    const CausalSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size       = 1,
        .key_position_map = g_attention_key_position_map,
        .resident_slots   = g_attention_key_slots,
    };
    const PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_small_t_launch_for<CausalD256H24Kv4>(q, input, pos, scale, batch_cache,
                                                              invocation, envelope, partial_acc,
                                                              partial_m, partial_l, out, stream);
        return;
    }
    causal_attention_small_t_launch_for<CausalD256H16Kv2>(q, input, pos, scale, batch_cache,
                                                          invocation, envelope, partial_acc,
                                                          partial_m, partial_l, out, stream);
}

} // namespace ninfer::ops::detail
