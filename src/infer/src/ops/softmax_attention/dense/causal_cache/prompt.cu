// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16_wmma.cuh" // AMD port: gfx12 WMMA prompt path
#include "ops/kv_cache/q4_0_codec.cuh" // AMD port: q4_0 planes (q4_0_code_offset/scale_offset)
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

// AMD port (gfx1201): NINFER_ATTN_PROMPT_WMMA=0 forces the chunked small-T fallback.
// The WMMA prompt kernel needs only 40 KiB of dynamic smem (under the 48 KiB default),
// so unlike the upstream 96 KiB prompt kernel it actually launches on gfx12.
[[nodiscard]] bool attn_prompt_wmma_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_ATTN_PROMPT_WMMA");
        return v == nullptr || std::string(v) != "0";
    }();
    return enabled;
}

// AMD port (gfx1201): T=1 decode split-KV WMMA kernel gate. Default ON (the
// alternative for bf16/fp16/q4_0 is the prefill WMMA kernel with 63/64 padding
// rows, which at long context measured 2.3 t/s and produced NaN at @100k; the
// split-KV kernel measures 11.8 t/s and stays finite). Disable with
// NINFER_DECODE_WMMA=0.
[[nodiscard]] bool decode_t1_wmma_enabled() {
    static const bool enabled = [] {
        // cmd's `set VAR=1 &&` keeps the trailing space in the value; accept any
        // value whose first non-space char is '1' (0 / unset stay off).
        const char* v = std::getenv("NINFER_DECODE_WMMA");
        bool on = true;
        if (v != nullptr) {
            while (*v == ' ' || *v == '\t') { ++v; }
            on = *v != '0';
        }
        return on;
    }();
    return enabled;
}

// AMD port (gfx1201): the KInt8 WMMA prompt kernel needs 57856 B dynamic smem (Vt + Ps
// + staged rotated-Q int8 codes + G64 scales), above the 48 KiB default ceiling but
// under the gfx12 64 KiB per-block cap; the attribute is set once per instantiation.
template <typename Geometry, typename Metadata>
void causal_attention_prompt_wmma_attr_kint8() {
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, false, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptWmmaInt8SmemBytes);
    CUDA_CHECK(attr);
}

// AMD port (gfx1201): RotatedInt4KeyInt4ValueE8 reuses the Int8Group64 staging layout
// (rotated Q codes + G64 scales live in the same smem tail), so the same 57856 B
// dynamic-smem request applies; the attribute is set once per instantiation.
// AMD port (gfx1201): KInt8Bf16 twin - int8 G64 codes dequantized to bf16 (no
// mma_s8_hw_direct). Same 57856 B dynamic-smem request as the native s8 variant.
[[nodiscard]] bool kint8_bf16_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_KINT8_BF16");
        return v != nullptr && v[0] != '0';
    }();
    return enabled;
}

template <typename Geometry, typename Metadata>
void causal_attention_prompt_wmma_attr_kint8b() {
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, false, false, true, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptWmmaInt8SmemBytes);
    CUDA_CHECK(attr);
}

template <typename Geometry, typename Metadata>
void causal_attention_prompt_wmma_attr_kint4() {
    static const cudaError_t attr = cudaFuncSetAttribute(
        causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, false, false, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptWmmaInt8SmemBytes);
    CUDA_CHECK(attr);
}

template <typename Geometry, typename CacheView, typename Metadata, bool KFp16 = false,
          bool KQ4 = false, bool KInt8 = false, bool KInt4 = false, bool KInt8Bf16 = false>
void causal_attention_prompt_wmma_launch_for(const Tensor& q, const Tensor& positions,
                                             float scale, const CacheView& cache,
                                             Metadata metadata, Tensor& out,
                                             cudaStream_t stream) {
    if constexpr (KInt8) { causal_attention_prompt_wmma_attr_kint8<Geometry, Metadata>(); }
    if constexpr (KInt8Bf16) { causal_attention_prompt_wmma_attr_kint8b<Geometry, Metadata>(); }
    if constexpr (KInt4) { causal_attention_prompt_wmma_attr_kint4<Geometry, Metadata>(); }
    const dim3 grid(static_cast<unsigned>(div_up(q.ne[2], kCausalPromptBr)),
                    static_cast<unsigned>(Geometry::QHeads), 1u);
    constexpr int smem_bytes = (KInt8 || KInt4 || KInt8Bf16) ? kCausalPromptWmmaInt8SmemBytes
                                                             : kCausalPromptWmmaSmemBytes;
    auto* k_scale =
        cache.k_scale_pages.data != nullptr ? static_cast<const __half*>(cache.k_scale_pages.data)
                                            : nullptr;
    auto* v_scale =
        cache.v_scale_pages.data != nullptr ? static_cast<const __half*>(cache.v_scale_pages.data)
                                            : nullptr;
    causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, KFp16, KQ4, KInt8, KInt4,
                                             KInt8Bf16><<<
        grid, kCausalPromptThreads, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(cache.k_pages.data),
        static_cast<const __half*>(cache.v_pages.data), k_scale, v_scale, metadata,
        static_cast<const std::int32_t*>(positions.data), scale,
        static_cast<__nv_bfloat16*>(out.data), q.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    // AMD port (gfx1201): the 96 KiB request exceeds the gfx12 64 KiB block cap — skip the
    // attribute (the upstream kernels are never launched here; the WMMA kernel below needs
    // only the default 48 KiB ceiling).
    if (kCausalPromptSmemBytes <= 64 * 1024) {
        static const cudaError_t attr_bf16 =
            cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
        CUDA_CHECK(attr_bf16);
    }
    if (kCausalPromptI8SmemBytes <= 64 * 1024) {
        static const cudaError_t attr_i8 =
            cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
        CUDA_CHECK(attr_i8);
    }

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.storage == KvCacheStorage::Int8Group64) {
        // AMD port (gfx1201): Int8Group64 prefill rides the KInt8 WMMA prompt kernel
        // (rotated-Q staging + G64 int8 dequant). The upstream i8 kernel requests
        // 92672 B smem (>64 KiB cap) and mma_s8 traps on gfx1201, so it can never
        // launch here.
        causal_attention_prompt_wmma_attr_kint8<Geometry, Metadata>();
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        auto* k_scale = cache.k_scale_pages.data != nullptr
                            ? static_cast<const __half*>(cache.k_scale_pages.data)
                            : nullptr;
        auto* v_scale = cache.v_scale_pages.data != nullptr
                            ? static_cast<const __half*>(cache.v_scale_pages.data)
                            : nullptr;
        if (kint8_bf16_enabled()) {
            causal_attention_prompt_wmma_attr_kint8b<Geometry, Metadata>();
            causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, false, false, true, false,
                                                     true>
                <<<attention_grid, kCausalPromptThreads, kCausalPromptWmmaInt8SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __half*>(cache_v.data), k_scale, v_scale, metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        } else {
            causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, false, false, true>
                <<<attention_grid, kCausalPromptThreads, kCausalPromptWmmaInt8SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __half*>(cache_v.data), k_scale, v_scale, metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        }
    } else {
        std::fprintf(stderr, "[wmma-fallback] q_heads=%d kv_heads=%d tokens=%d storage=%d\n",
                     Geometry::QHeads, Geometry::KVHeads, tokens, static_cast<int>(cache.storage));
        std::fflush(stderr);
        // AMD port (gfx1201): the upstream bf16 prompt kernel requests 96 KiB dynamic smem,
        // above the gfx12 64 KiB cap — it can never launch here. Route this geometry through
        // the WMMA kernel (40 KiB, default smem ceiling).
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        cudaError_t werr = cudaGetLastError();  // catch pending error from earlier ops
        if (werr != cudaSuccess) {
            std::fprintf(stderr, "[wmma-pending] pending error BEFORE launch: %s (qh=%d tokens=%d)\n",
                         cudaGetErrorString(werr), Geometry::QHeads, tokens);
            std::fflush(stderr);
        }
        auto* k_scale = cache.k_scale_pages.data != nullptr
                    ? static_cast<const __half*>(cache.k_scale_pages.data)
                    : nullptr;
        auto* v_scale = cache.v_scale_pages.data != nullptr
                            ? static_cast<const __half*>(cache.v_scale_pages.data)
                            : nullptr;
        if (cache.storage == KvCacheStorage::Q4_0) {
            // q4_0 paged-KV storage: read fp16 K via the fp16 WMMA twin.
            causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, true, true>
                <<<attention_grid, kCausalPromptThreads, kCausalPromptWmmaSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __half*>(cache_v.data), k_scale, v_scale, metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        } else if (cache.storage == KvCacheStorage::Fp16) {
            // Fp16 paged-KV storage: read fp16 K via the fp16 WMMA twin.
            causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata, true, false>
                <<<attention_grid, kCausalPromptThreads, kCausalPromptWmmaSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __half*>(cache_v.data), k_scale, v_scale, metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        } else {
            causal_attention_prompt_bf16_wmma_kernel<Geometry, Metadata>
                <<<attention_grid, kCausalPromptThreads, kCausalPromptWmmaSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(cache_k.data),
                    static_cast<const __half*>(cache_v.data), nullptr, nullptr, metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        }
        werr = cudaGetLastError();
        if (werr != cudaSuccess) {
            std::fprintf(stderr, "[wmma-launch-fail] grid=%u,%u,%u tokens=%d qh=%d err=%s\n",
                         attention_grid.x, attention_grid.y, attention_grid.z, tokens,
                         Geometry::QHeads, cudaGetErrorString(werr));
            std::fflush(stderr);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream) {
    if ((cache.storage == KvCacheStorage::BFloat16 || cache.storage == KvCacheStorage::Fp16 ||
         cache.storage == KvCacheStorage::Q4_0 ||
         cache.storage == KvCacheStorage::Int8Group64 ||
         cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) &&
        attn_prompt_wmma_enabled()) {
        const PagedKVDirectMetadata wmma_metadata{
            static_cast<const std::int32_t*>(cache.block_table.data)};
        const auto launch_q = [&]<bool KFp16, bool KQ4, bool KInt8, bool KInt4, bool KInt8Bf16>() {
            if (q.ne[1] == CausalD256H24Kv4::QHeads) {
                causal_attention_prompt_wmma_launch_for<
                    CausalD256H24Kv4, PagedKVLayerView, PagedKVDirectMetadata, KFp16, KQ4, KInt8,
                    KInt4, KInt8Bf16>(q, positions, scale, cache, wmma_metadata, out, stream);
            } else {
                causal_attention_prompt_wmma_launch_for<
                    CausalD256H16Kv2, PagedKVLayerView, PagedKVDirectMetadata, KFp16, KQ4, KInt8,
                    KInt4, KInt8Bf16>(q, positions, scale, cache, wmma_metadata, out, stream);
            }
        };
        if (cache.storage == KvCacheStorage::Q4_0) {
            launch_q.template operator()<true, true, false, false, false>();
        } else if (cache.storage == KvCacheStorage::Fp16) {
            launch_q.template operator()<true, false, false, false, false>();
        } else if (cache.storage == KvCacheStorage::Int8Group64) {
            // AMD port (gfx1201): NINFER_KINT8_BF16=1 swaps the native s8 QK for the
            // dequant-to-bf16 twin (no mma_s8_hw_direct) without changing the KV bytes.
            if (kint8_bf16_enabled()) {
                launch_q.template operator()<false, false, true, false, true>();
            } else {
                launch_q.template operator()<false, false, true, false, false>();
            }
        } else if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
            launch_q.template operator()<false, false, false, true, false>();
        } else {
            launch_q.template operator()<false, false, false, false, false>();
        }
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    // AMD port: the speculative VERIFY pass arrives with a non-null valid_columns mask and
    // is therefore routed through PagedKVBatchMetadata<true>, whose block_table() resolves
    // the page table as tables + table_rows[0]*table_stride (the direct metadata uses row
    // 0). A single-sequence verify has exactly one table row, so the two are the same
    // thing -- and the direct form is the one probe/attn_flash_probe.cu --sweep and
    // --positions actually cover (T=1..80, pos0=512).
    const bool direct_ok =
        valid_columns.data == nullptr || (q.ne[2] <= 16 && q.ne[3] == 1);
    if ((cache.storage == KvCacheStorage::BFloat16 || cache.storage == KvCacheStorage::Fp16 ||
         cache.storage == KvCacheStorage::Q4_0 ||
         cache.storage == KvCacheStorage::Int8Group64 ||
         cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) &&
        direct_ok && attn_prompt_wmma_enabled()) {
        // Batch size 1 (prefill) has a single page table row; point the direct
        // metadata at it, mirroring PagedKVBatchMetadata::block_table().
        const PagedKVDirectMetadata wmma_metadata{
            static_cast<const std::int32_t*>(cache.block_tables.data)};
        const auto launch_q = [&]<bool KFp16, bool KQ4, bool KInt8, bool KInt4, bool KInt8Bf16>() {
            if (q.ne[1] == CausalD256H24Kv4::QHeads) {
                causal_attention_prompt_wmma_launch_for<
                    CausalD256H24Kv4, PagedKVBatchLayerView, PagedKVDirectMetadata, KFp16, KQ4,
                    KInt8, KInt4, KInt8Bf16>(q, positions, scale, cache, wmma_metadata, out, stream);
            } else {
                causal_attention_prompt_wmma_launch_for<
                    CausalD256H16Kv2, PagedKVBatchLayerView, PagedKVDirectMetadata, KFp16, KQ4,
                    KInt8, KInt4, KInt8Bf16>(q, positions, scale, cache, wmma_metadata, out, stream);
            }
        };
        if (cache.storage == KvCacheStorage::Q4_0) {
            launch_q.template operator()<true, true, false, false, false>();
        } else if (cache.storage == KvCacheStorage::Fp16) {
            launch_q.template operator()<true, false, false, false, false>();
        } else if (cache.storage == KvCacheStorage::Int8Group64) {
            // AMD port (gfx1201): NINFER_KINT8_BF16=1 swaps the native s8 QK for the
            // dequant-to-bf16 twin (no mma_s8_hw_direct) without changing the KV bytes.
            if (kint8_bf16_enabled()) {
                launch_q.template operator()<false, false, true, false, true>();
            } else {
                launch_q.template operator()<false, false, true, false, false>();
            }
        } else if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
            launch_q.template operator()<false, false, false, true, false>();
        } else {
            launch_q.template operator()<false, false, false, false, false>();
        }
        return;
    }
    const auto launch = [&]<bool Masked>() {
        // AMD port diagnostic: the masked metadata resolves its page table as
        // tables + table_rows[0]*table_stride, while the direct one just uses row 0.
        // If table_stride (ne[0]) is not the real rows-per-batch stride, every masked
        // launch reads the WRONG page table -- which is exactly the speculative verify
        // path, and it would look like "the verify pass is wrong above width 8".
        static bool meta_printed = false;
        if (!meta_printed) {
            meta_printed = true;
            std::fprintf(stderr, "[prompt-meta] masked=%d bt_ne0=%lld bt_ne1=%lld bt_stride=%lld\n",
                         Masked ? 1 : 0, static_cast<long long>(cache.block_tables.ne[0]),
                         static_cast<long long>(cache.block_tables.ne[1]),
                         static_cast<long long>(cache.block_tables.nb[1] / 4));
            std::fflush(stderr);
        }
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail


