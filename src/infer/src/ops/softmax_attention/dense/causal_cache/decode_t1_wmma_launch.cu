// AMD port (gfx1201): T=1 decode attention via the zero-shuffle WMMA split-KV kernel
// (decode_t1_wmma.cuh) + the production small-T reducer. The prefill prompt WMMA
// kernel would waste 63/64 of its query rows at width=1; the split-KV kernel reads
// the paged KV at full bandwidth (decode attention measured 93% of the decode step
// at 100k ctx, so this is the release-critical path).
#include "core/arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/decode_t1_wmma.cuh"
#include "ops/softmax_attention/dense/causal_cache/decode_t1_simt.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/device.h" // CUDA_CHECK

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kCausalDecodeHeadDim = 256;  // kCausalHeadDim

// Partials in the layout the production reducer consumes (acc [D, QHeads, tokens,
// splits], m/l [QHeads, tokens, splits]).
Tensor allocate_acc(WorkspaceArena& workspace, std::int32_t q_heads, std::int32_t splits) {
    // +8 floats tail for NINFER_ATTN_PROFILE stage-cycle accumulation.
    return workspace.alloc(DType::FP32, {kCausalDecodeHeadDim, q_heads, 1, splits + 8});
}
Tensor allocate_stat(WorkspaceArena& workspace, std::int32_t q_heads, std::int32_t splits) {
    return workspace.alloc(DType::FP32, {q_heads, 1, splits});
}

template <typename CacheInput, bool KFp16, bool KQ4, bool KFp8 = false, bool KInt4E8 = false,
          bool KInt8 = false>
void launch_partials(std::int32_t q_heads, std::int32_t splits, std::int32_t kv_heads,
                     const Tensor& q, const Tensor& q_stage, const Tensor& positions,
                     PagedKVBatchLayerView cache, const std::int32_t* tab_rows,
                     std::int32_t table_stride, float scale, CacheInput input,
                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out,
                     std::int32_t token_base, cudaStream_t stream) {
    // AMD port: SIMT flash-attention decode path (NINFER_ATTN_SIMT=1). Emits the same
    // partials as the WMMA kernel; the production reduce kernel below consumes them.
    // FP8 / rk4v4-e8 KV have no WMMA decoder here, so they force the SIMT path as well.
    // Default ON since 2026-10-27. On gfx1201 the SIMT decode route beats the WMMA one at
    // 32K bf16 (attention 11.04 -> 4.66 ms/step) because its grid is q_heads/2 x splits = 192
    // CTAs where the WMMA route's kv_heads x splits/4 is far fewer.
    // NINFER_ATTN_SIMT=0 restores the WMMA path (the only one that honours NINFER_SKIP_ATTN).
    //
    // KInt8 (Int8Group64) has no SIMT decoder, so it always rides the WMMA route.
    if (std::getenv("NINFER_DBG_ADDR") != nullptr) {
        static bool addr_printed = false;
        if (!addr_printed) {
            addr_printed = true;
            std::fprintf(stderr,
                         "[addr] q=%p qn=%lld q_stage=%p psn=%lld acc=%p accn=%lld m=%p mn=%lld "
                         "l=%p ln=%lld out=%p outn=%lld token_base=%d\n",
                         (const void*)q.data, (long long)q.numel(), (const void*)q_stage.data,
                         (long long)q_stage.numel(), (const void*)partial_acc.data,
                         (long long)partial_acc.numel(), (const void*)partial_m.data,
                         (long long)partial_m.numel(), (const void*)partial_l.data,
                         (long long)partial_l.numel(), (const void*)out.data,
                         (long long)out.numel(), token_base);
            std::fflush(stderr);
        }
    }
    static const bool simt_on = [] {
        const char* e = std::getenv("NINFER_ATTN_SIMT");
        return e == nullptr || e[0] != '0';
    }();
    // SIMT decode for bf16 / fp16 / q4_0 / fp8; int8 and rk4v4-e8 ride WMMA.
    // EXPERIMENT 2026-10-30 #5 (REVERTED): routing rk4v4-e8 into the WMMA kernel made it emit
    // garbage ("!!!!!!!!!!!!" at 8K) and run slower (31.2 vs 36.0 t/s) -- the KInt4E8 WMMA
    // instantiation had never been exercised and its tile dequant is not implemented. int4
    // therefore stays on SIMT.
    if ((simt_on && !KInt8) || KFp8) {
        constexpr int kSimtThreadsC = decode_t1_simt_detail::kSimtThreads;
        // split capacity is the engine's (matches partial_acc allocation); the SIMT
        // kernel resolves window/active splits on device, exactly like the WMMA kernel.
        // grid.x covers q_heads/2 CTAs, each handling 2 q rows sharing one kv head.
        // NINFER_SIMT_SPLITS overrides the split count (partial slots allocated for the
        // engine's splits are large enough; the SIMT launch just uses fewer).
        static const int simt_splits_override = [] {
            const char* e = std::getenv("NINFER_SIMT_SPLITS");
            return e != nullptr ? std::atoi(e) : 0;
        }();
        const std::int32_t launch_splits =
            simt_splits_override > 0 ? std::min(simt_splits_override, splits) : splits;
        static bool sp_printed = false;
        if (!sp_printed) {
            sp_printed = true;
            const char* dv_env = std::getenv("NINFER_DECODE_WMMA");
            std::fprintf(stderr, "[simt-splits] engine_splits=%d override=%d launch=%d decwmma=%s mfix=scaled\n",
                         splits, simt_splits_override, launch_splits,
                         dv_env == nullptr ? "unset" : dv_env);
            std::fflush(stderr);
        }
        const dim3 sg(q_heads / 2, launch_splits, 1);
        // gfx1201 full-group variant: one CTA per (kv_head, split) covering EVERY q row
        // that shares that kv head (GroupSize = q_heads / kv_heads). The cached K/V row is
        // then decoded+loaded once for GroupSize rows instead of once per 2-row CTA, which
        // removes (GroupSize/2)-fold redundant dequant / load work from the decode step.
        const dim3 sg_group(kv_heads, launch_splits, 1);
        constexpr int kGroupRows = CausalD256H24Kv4::QHeads / CausalD256H24Kv4::KVHeads;
        auto* k_scale = cache.k_scale_pages.data != nullptr
                            ? static_cast<__half*>(cache.k_scale_pages.data)
                            : nullptr;
        auto* v_scale = cache.v_scale_pages.data != nullptr
                            ? static_cast<__half*>(cache.v_scale_pages.data)
                            : nullptr;
        // NINFER_SIMT_GROUP=0 switches to the 2-row CTA variant: same math, but each
        // thread holds qr[2][8] + q_rot[2][8] + o[2][8] instead of the 6-row set, which
        // cuts register pressure ~3x and lifts occupancy (the 6-row variant pins 1 CTA/SM
        // because __launch_bounds__ allows 256 regs/thread). It reads each KV row 3x more
        // often, but the decode attention path is latency-bound, not bandwidth-bound
        // (bf16 and int8 KV run at the same speed), so this trades free bytes for occupancy.
        // DEFAULT (measured 2026-10-30): int4 KV takes the 2-row CTA, every wider format keeps
        // the full q-head group.
        //   int4 rk4v4-e8, 8 warps/CTA: 6-row -> 48K decode 28.36, 32K 32.25;
        //                               2-row -> 48K decode 30.70, 32K 33.62 (+8.3% / +4.2%).
        //   An int4 K/V row pair is only ~280B, so the 3x redundant reads of a 2-row CTA land in
        //   L2 (867MB unique at 48K) and cost almost nothing, while tripling the CTA count lifts
        //   resident warps from 16 to 32 per CU - exactly what this latency-bound kernel needs.
        //   bf16 is the opposite: rows are 1024B, the 3x duplicate stream does NOT fit L2, and
        //   2-row measured 10.6 t/s at 32K against 31.5 for the 6-row variant.
        static const bool simt_group = [] {
            const char* e = std::getenv("NINFER_SIMT_GROUP");
            if (e != nullptr) { return e[0] != '0'; }
            return !KInt4E8;
        }();
        // NINFER_SKIP_ATTN_SIMT=1 makes the SIMT partial kernel write neutral partials and
        // return, which isolates the K/V read + online softmax cost from the projections.
        static const int simt_skip_attn = [] {
            if (std::getenv("NINFER_SIMT_DBGP") != nullptr) { return 2; }
            const char* e = std::getenv("NINFER_SKIP_ATTN_SIMT");
            return (e != nullptr && e[0] != '0') ? 1 : 0;
        }();
        if (std::getenv("NINFER_ATTN_OCC") != nullptr) {
            static bool occ_printed = false;
            if (!occ_printed) {
                occ_printed = true;
                int occ_group = 0;
                int occ_pair  = 0;

                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occ_group,
                    causal_decode_t1_simt_partial_kernel<CausalD256H24Kv4, CacheInput, KFp16, KQ4,
                                                         KFp8, KInt4E8, kGroupRows>,
                    kSimtThreadsC, 0);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occ_pair,
                    causal_decode_t1_simt_partial_kernel<CausalD256H24Kv4, CacheInput, KFp16, KQ4,
                                                         KFp8, KInt4E8, 2>,
                    kSimtThreadsC, 0);
                std::fprintf(stderr, "[attn-occ] group_rows CTAs/SM=%d  pair_rows CTAs/SM=%d\n",
                             occ_group, occ_pair);
                std::fflush(stderr);
            }
        }
        {
            // Probe: zero the partial buffers before the kernel so that "value differs after the
            // kernel" can only mean the kernel itself produced it (not stale arena contents).
            static const bool acc_zero = [] {
                const char* e = std::getenv("NINFER_ACC_ZERO");
                return e != nullptr && e[0] == '1';
            }();
            if (acc_zero) {
                const std::int64_t nz =
                    static_cast<std::int64_t>(CausalD256H24Kv4::QHeads) * launch_splits;
                CUDA_CHECK(cudaMemsetAsync(partial_acc.data, 0,
                                           static_cast<size_t>(nz) * 256 * sizeof(float), stream));
                CUDA_CHECK(cudaMemsetAsync(partial_m.data, 0,
                                           static_cast<size_t>(nz) * sizeof(float), stream));
                CUDA_CHECK(cudaMemsetAsync(partial_l.data, 0,
                                           static_cast<size_t>(nz) * sizeof(float), stream));
            }
        }
        if (q_heads == CausalD256H24Kv4::QHeads) {
            if (simt_group) {
                causal_decode_t1_simt_partial_kernel<CausalD256H24Kv4, CacheInput, KFp16, KQ4,
                                                     KFp8, KInt4E8, kGroupRows>
                    <<<sg_group, kSimtThreadsC, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data), input,
                        static_cast<__nv_bfloat16*>(cache.k_pages.data),
                        static_cast<__half*>(cache.v_pages.data), k_scale, v_scale,
                        static_cast<const std::int32_t*>(cache.block_tables.data), tab_rows,
                        table_stride, static_cast<const std::int32_t*>(positions.data), scale,
                        static_cast<float*>(partial_acc.data),
                        static_cast<float*>(partial_m.data),
                        static_cast<float*>(partial_l.data), simt_skip_attn, token_base);
            } else {
                causal_decode_t1_simt_partial_kernel<CausalD256H24Kv4, CacheInput, KFp16, KQ4,
                                                     KFp8, KInt4E8, 2>
                    <<<sg, kSimtThreadsC, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data), input,
                        static_cast<__nv_bfloat16*>(cache.k_pages.data),
                        static_cast<__half*>(cache.v_pages.data), k_scale, v_scale,
                        static_cast<const std::int32_t*>(cache.block_tables.data), tab_rows,
                        table_stride, static_cast<const std::int32_t*>(positions.data), scale,
                        static_cast<float*>(partial_acc.data),
                        static_cast<float*>(partial_m.data),
                        static_cast<float*>(partial_l.data), simt_skip_attn, token_base);
            }
            {
                static const bool pdbg_on = [] {
                    const char* e = std::getenv("NINFER_PDBG");
                    return e != nullptr && e[0] == '1';
                }();
                static int pdbg_n = 0;
                if (pdbg_on && pdbg_n < 4) {
                    cudaStreamCaptureStatus pcap = cudaStreamCaptureStatusNone;
                    CUDA_CHECK(cudaStreamIsCapturing(stream, &pcap));
                    if (pcap == cudaStreamCaptureStatusNone) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        const std::int64_t nstat =
                            static_cast<std::int64_t>(CausalD256H24Kv4::QHeads) * (launch_splits + 8);
                        std::vector<float> hm(static_cast<size_t>(nstat));
                        std::vector<float> hl(static_cast<size_t>(nstat));
                        std::vector<float> ha(static_cast<size_t>(nstat) * 256);
                        CUDA_CHECK(cudaMemcpy(hm.data(), partial_m.data, hm.size() * sizeof(float),
                                              cudaMemcpyDeviceToHost));  // [split][head], tail unused
                        CUDA_CHECK(cudaMemcpy(hl.data(), partial_l.data, hl.size() * sizeof(float),
                                              cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(ha.data(), partial_acc.data, ha.size() * sizeof(float),
                                              cudaMemcpyDeviceToHost));
                        double sm = 0.0, sl = 0.0, sa = 0.0;
                        for (float v : hm) sm += v;
                        for (float v : hl) sl += v;
                        for (float v : ha) sa += static_cast<double>(v);
                        const std::size_t kProbe = 12u << 20;
                        std::vector<unsigned char> hv(kProbe, 0);
                        std::vector<unsigned char> hk(kProbe, 0);
                        std::vector<unsigned char> hvs(1u << 21, 0);
                        std::vector<unsigned char> hks(1u << 21, 0);
                        std::vector<int> htab(static_cast<size_t>(cache.block_tables.numel()), 0);
                        const std::size_t kTabCopy =
                            std::min<size_t>(htab.size(), static_cast<size_t>(1) << 20) * sizeof(int);
                        if (cache.v_pages.data != nullptr) {
                            CUDA_CHECK(cudaMemcpy(hv.data(), cache.v_pages.data, kProbe,
                                                  cudaMemcpyDeviceToHost));
                        }
                        if (cache.k_pages.data != nullptr) {
                            CUDA_CHECK(cudaMemcpy(hk.data(), cache.k_pages.data, kProbe,
                                                  cudaMemcpyDeviceToHost));
                        }
                        if (cache.v_scale_pages.data != nullptr) {
                            CUDA_CHECK(cudaMemcpy(hvs.data(), cache.v_scale_pages.data, 1u << 21,
                                                  cudaMemcpyDeviceToHost));
                        }
                        if (cache.k_scale_pages.data != nullptr) {
                            CUDA_CHECK(cudaMemcpy(hks.data(), cache.k_scale_pages.data, 1u << 21,
                                                  cudaMemcpyDeviceToHost));
                        }
                        if (cache.block_tables.data != nullptr && kTabCopy > 0) {
                            CUDA_CHECK(cudaMemcpy(htab.data(), cache.block_tables.data, kTabCopy,
                                                  cudaMemcpyDeviceToHost));
                        }
                        std::vector<int> hpos(static_cast<size_t>(positions.numel()), 0);
                        if (positions.data != nullptr && !hpos.empty()) {
                            const size_t npos =
                                std::min<size_t>(hpos.size(), static_cast<size_t>(1) << 16);
                            CUDA_CHECK(cudaMemcpy(hpos.data(), positions.data, npos * sizeof(int),
                                                  cudaMemcpyDeviceToHost));
                        }
                        std::fprintf(stderr, "[pdbg-pos] n=%d tok=%d scale=%.9g pos0=%d pos1=%d pos2=%d\n",
                                     pdbg_n, token_base, (double)scale,
                                     hpos.empty() ? -1 : hpos[0],
                                     hpos.size() < 2 ? -1 : hpos[1],
                                     hpos.size() < 3 ? -1 : hpos[2]);
                        std::fflush(stderr);
                        std::vector<unsigned char> hq(static_cast<size_t>(q.numel()) * 2, 0);
                        if (q.data != nullptr && !hq.empty()) {
                            CUDA_CHECK(cudaMemcpy(hq.data(), q.data, hq.size(),
                                                  cudaMemcpyDeviceToHost));
                        }
                        char fn[256];
                        std::snprintf(fn, sizeof(fn), "D:/ninfer-hip-migrate/build/_pd_%d.bin", pdbg_n);
                        FILE* pf = std::fopen(fn, "wb");
                        if (pf != nullptr) {
                            std::fwrite(hm.data(), sizeof(float), hm.size(), pf);
                            std::fwrite(hl.data(), sizeof(float), hl.size(), pf);
                            std::fwrite(ha.data(), sizeof(float), ha.size(), pf);
                            std::fwrite(hv.data(), 1, kProbe, pf);
                            std::fwrite(hk.data(), 1, kProbe, pf);
                            std::fwrite(hvs.data(), 1, 1u << 21, pf);
                            std::fwrite(hks.data(), 1, 1u << 21, pf);
                            std::fwrite(htab.data(), 1, kTabCopy, pf);
                            std::fwrite(hpos.data(), 1, hpos.size() * sizeof(int), pf);
                            std::fwrite(hq.data(), 1, hq.size(), pf);
                            std::fclose(pf);
                        }
                        double h0 = 0.0, h10 = 0.0;
                        for (int d = 0; d < 256; ++d) {
                            h0 += ha[static_cast<size_t>(d)];
                            h10 += ha[(10 + static_cast<std::size_t>(0) * 0) * 256 + static_cast<size_t>(d)];
                        }
                        std::fprintf(stderr,
                                     "[pdbg] n=%d splits=%d sumM=%.9g sumL=%.9g sumA=%.9g h0=%.9g h10=%.9g\n",
                                     pdbg_n, launch_splits, sm, sl, sa, h0, h10);
                        std::fflush(stderr);
                        ++pdbg_n;
                    }
                }
            }
            constexpr int kDChunkS = 256;
            const dim3 rs(CausalD256H24Kv4::QHeads, div_up(kCausalHeadDim, kDChunkS), 1);
            // AMD port: token_base selects which column of the multi-token verify batch this
            // (partial, reduce) pair belongs to. Offset=true shifts the positions pointer by
            // column_begin, so the reducer sees a 1-token window and writes column
            // column_begin — exactly the T=1 contract, repeated per column.
            if (token_base == 0) {
                causal_attention_small_t_reduce_output_kernel<
                    CausalD256H24Kv4, kDChunkS, false, false, false, false>
                    <<<rs, 256, 0, stream>>>(
                        static_cast<const float*>(partial_acc.data),
                        static_cast<const float*>(partial_m.data),
                        static_cast<const float*>(partial_l.data),
                        static_cast<const std::int32_t*>(positions.data), nullptr, 1, 1, 0, 1,
                        launch_splits, static_cast<__nv_bfloat16*>(out.data));
                static const bool qdbg_on = [] {
                    const char* e = std::getenv("NINFER_PDBG");
                    return e != nullptr && e[0] == '1';
                }();
                static int qdbg_n = 0;
                if (qdbg_on && qdbg_n < 4) {
                    cudaStreamCaptureStatus qcap = cudaStreamCaptureStatusNone;
                    CUDA_CHECK(cudaStreamIsCapturing(stream, &qcap));
                    if (qcap == cudaStreamCaptureStatusNone) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        const std::int64_t nstat2 =
                            static_cast<std::int64_t>(CausalD256H24Kv4::QHeads) * launch_splits;
                        std::vector<float> hq(static_cast<size_t>(nstat2) * 256);
                        CUDA_CHECK(cudaMemcpy(hq.data(), partial_acc.data,
                                              hq.size() * sizeof(float),
                                              cudaMemcpyDeviceToHost));
                        char qfn[256];
                        std::snprintf(qfn, sizeof(qfn),
                                      "D:/ninfer-hip-migrate/build/_pq_%d.bin", qdbg_n);
                        FILE* qf = std::fopen(qfn, "wb");
                        if (qf != nullptr) {
                            std::fwrite(hq.data(), sizeof(float), hq.size(), qf);
                            std::fclose(qf);
                        }
                        ++qdbg_n;
                    }
                }
            } else {
                causal_attention_small_t_reduce_output_kernel<
                    CausalD256H24Kv4, kDChunkS, false, false, false, true>
                    <<<rs, 256, 0, stream>>>(
                        static_cast<const float*>(partial_acc.data),
                        static_cast<const float*>(partial_m.data),
                        static_cast<const float*>(partial_l.data),
                        static_cast<const std::int32_t*>(positions.data), nullptr, 1, 1, token_base,
                        1, launch_splits, static_cast<__nv_bfloat16*>(out.data));
            }
        } else {
            causal_decode_t1_simt_partial_kernel<CausalD256H16Kv2, CacheInput, KFp16, KQ4, KFp8,
                                                 KInt4E8>
                <<<sg, kSimtThreadsC, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), input,
                    static_cast<__nv_bfloat16*>(cache.k_pages.data),
                    static_cast<__half*>(cache.v_pages.data), k_scale, v_scale,
                    static_cast<const std::int32_t*>(cache.block_tables.data), tab_rows,
                    table_stride, static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<float*>(partial_acc.data),
                    static_cast<float*>(partial_m.data),
                    static_cast<float*>(partial_l.data), simt_skip_attn, token_base);
            constexpr int kDChunkH16S = 64;
            const dim3 rs(CausalD256H16Kv2::QHeads, div_up(kCausalHeadDim, kDChunkH16S), 1);
            if (token_base == 0) {
                causal_attention_small_t_reduce_output_kernel<
                    CausalD256H16Kv2, kDChunkH16S, false, false, false, false>
                    <<<rs, 256, 0, stream>>>(
                        static_cast<const float*>(partial_acc.data),
                        static_cast<const float*>(partial_m.data),
                        static_cast<const float*>(partial_l.data),
                        static_cast<const std::int32_t*>(positions.data), nullptr, 1, 1, 0, 1,
                        launch_splits, static_cast<__nv_bfloat16*>(out.data));
            } else {
                causal_attention_small_t_reduce_output_kernel<
                    CausalD256H16Kv2, kDChunkH16S, false, false, false, true>
                    <<<rs, 256, 0, stream>>>(
                        static_cast<const float*>(partial_acc.data),
                        static_cast<const float*>(partial_m.data),
                        static_cast<const float*>(partial_l.data),
                        static_cast<const std::int32_t*>(positions.data), nullptr, 1, 1, token_base,
                        1, launch_splits, static_cast<__nv_bfloat16*>(out.data));
            }
        }
        return;
    }
    const std::int32_t* pos = static_cast<const std::int32_t*>(positions.data);
    const std::int32_t cap  = static_cast<std::int32_t>(0x1000000);  // hard bound, see kernel
    // B22 per-warp splits (KInt8): each CTA's 4 warps own 4 different splits, so the
    // grid carries splits/4 CTAs; the kernel maps warp -> split internally.
    const dim3 grid(kv_heads, KInt8 ? div_up(splits, 4) : splits, 1);
    auto* k_scale =
        cache.k_scale_pages.data != nullptr ? static_cast<__half*>(cache.k_scale_pages.data) : nullptr;
    auto* v_scale =
        cache.v_scale_pages.data != nullptr ? static_cast<__half*>(cache.v_scale_pages.data) : nullptr;
    // B22 Q-staging scratch tile: [kv_heads][16 rows][256] int8 codes + scales
    // [kv_heads][16][4]. Only the KInt8 path consumes it.
    const std::int32_t q_stage_bytes =
        kv_heads * kDecodeWmmaWarpRows * (kDecodeWmmaD + kKVCacheInt8Groups * 2);
    std::int8_t* q_codes_g = nullptr;
    __half* q_scale_g      = nullptr;
    if constexpr (KInt8) {
        q_codes_g = static_cast<std::int8_t*>(q_stage.data);
        q_scale_g = reinterpret_cast<__half*>(static_cast<std::int8_t*>(q_stage.data) +
                                              kv_heads * kDecodeWmmaWarpRows * kDecodeWmmaD);
    }
    const int smem_bytes = kDecodeWmmaSmemBytes;
    // B24 diagnosis: NINFER_SKIP_ATTN=1 zeroes the attention partials instead of
    // running QK/PV, so only the GEMV/other pipeline cost remains. Measures how
    // much of the per-step wall-clock attention really is (m=0,l=0,acc=0 -> the
    // reducer's head_l=0 -> out=0, finite, nothing depends on the value).
    static const bool skip_attn = [] {
        const char* e = std::getenv("NINFER_SKIP_ATTN");
        return e != nullptr && e[0] == '1';
    }();
    const auto zero_partials = [&] {
        const std::size_t acc_bytes =
            static_cast<std::size_t>(kCausalDecodeHeadDim) * q_heads * 1 * splits * 4;
        const std::size_t stat_bytes = static_cast<std::size_t>(q_heads) * 1 * splits * 4;
        hipMemsetAsync(partial_acc.data, 0, acc_bytes, stream);
        hipMemsetAsync(partial_m.data, 0, stat_bytes, stream);
        hipMemsetAsync(partial_l.data, 0, stat_bytes, stream);
    };
    if (q_heads == CausalD256H24Kv4::QHeads) {
        if (skip_attn) {
            zero_partials();
        } else if constexpr (KInt8) {
            causal_decode_t1_q_stage_kernel<CausalD256H24Kv4>
                <<<dim3(kv_heads, 1, 1), kDecodeWmmaThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data), q_codes_g, q_scale_g);
        }
        if (!skip_attn) {
        causal_decode_t1_wmma_partial_kernel<CausalD256H24Kv4, CacheInput, KFp16, KQ4, KInt4E8,
                                             KInt8>
            <<<grid, kDecodeWmmaThreads, smem_bytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), q_codes_g, q_scale_g, input, pos,
                static_cast<__nv_bfloat16*>(cache.k_pages.data),
                static_cast<__half*>(cache.v_pages.data), k_scale, v_scale,
                static_cast<const std::int32_t*>(cache.block_tables.data), tab_rows, table_stride,
                cap, splits, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data),
                static_cast<float*>(partial_l.data));
        }
        constexpr int kDChunk = 256;
        const dim3 rg(CausalD256H24Kv4::QHeads, div_up(kCausalHeadDim, kDChunk), 1);
        causal_attention_small_t_reduce_output_kernel<
            CausalD256H24Kv4, kDChunk, false, false, false, false>
            <<<rg, 256, 0, stream>>>(
                static_cast<const float*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data), pos, nullptr, 1, 1, 0, 1, splits,
                static_cast<__nv_bfloat16*>(out.data));
#ifdef NINFER_ATTN_PROFILE
        {  // read back attention stage cycles (host in non-graph runs)
            cudaStreamSynchronize(stream);
            float t[4] = {0};
            cudaMemcpy(t,
                       static_cast<const float*>(partial_acc.data) +
                           static_cast<std::size_t>(kCausalDecodeHeadDim) * q_heads * splits,
                       4 * sizeof(float), cudaMemcpyDeviceToHost);
            std::fprintf(stderr,
                         "[attn-prof] stage cycles: stage=%.0f qk=%.0f sm=%.0f pv=%.0f"
                         " total=%.2fms @1.7GHz\n",
                         t[0], t[1], t[2], t[3], (t[0] + t[1] + t[2] + t[3]) / 1.7e6);
        }
#endif
        return;
    }
    constexpr int kDChunkH16 = 64;
    if constexpr (KInt8) {
        causal_decode_t1_q_stage_kernel<CausalD256H16Kv2>
            <<<dim3(kv_heads, 1, 1), kDecodeWmmaThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), q_codes_g, q_scale_g);
    }
    causal_decode_t1_wmma_partial_kernel<CausalD256H16Kv2, CacheInput, KFp16, KQ4, KInt4E8, KInt8>
        <<<grid, kDecodeWmmaThreads, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), q_codes_g, q_scale_g, input, pos,
            static_cast<__nv_bfloat16*>(cache.k_pages.data),
            static_cast<__half*>(cache.v_pages.data), k_scale, v_scale,
            static_cast<const std::int32_t*>(cache.block_tables.data), tab_rows, table_stride,
            cap, splits, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    const dim3 rg(CausalD256H16Kv2::QHeads, div_up(kCausalHeadDim, kDChunkH16), 1);
    causal_attention_small_t_reduce_output_kernel<
        CausalD256H16Kv2, kDChunkH16, false, false, false, false>
        <<<rg, 256, 0, stream>>>(
            static_cast<const float*>(partial_acc.data),
            static_cast<const float*>(partial_m.data),
            static_cast<const float*>(partial_l.data), pos, nullptr, 1, 1, 0, 1, splits,
            static_cast<__nv_bfloat16*>(out.data));
}

template <typename CacheInput>
void launch_partials_for_storage(KvCacheStorage storage, std::int32_t q_heads,
                                 std::int32_t splits, std::int32_t kv_heads, const Tensor& q,
                                 const Tensor& q_stage, const Tensor& positions,
                                 PagedKVBatchLayerView cache, const std::int32_t* tab_rows,
                                 std::int32_t table_stride, float scale, CacheInput input,
                                 Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                 Tensor& out, std::int32_t token_base, cudaStream_t stream) {
    switch (storage) {
    case KvCacheStorage::Q4_0:
        launch_partials<CacheInput, true, true>(q_heads, splits, kv_heads, q, q_stage, positions,
                                                cache, tab_rows, table_stride, scale, input,
                                                partial_acc, partial_m, partial_l, out, token_base,
                                                stream);
        return;
    case KvCacheStorage::Fp16:
        launch_partials<CacheInput, true, false>(q_heads, splits, kv_heads, q, q_stage, positions,
                                                 cache, tab_rows, table_stride, scale, input,
                                                 partial_acc, partial_m, partial_l, out, token_base,
                                                 stream);
        return;
    case KvCacheStorage::Fp8E4M3Row256:
        launch_partials<CacheInput, false, false, true>(q_heads, splits, kv_heads, q, q_stage,
                                                        positions, cache, tab_rows, table_stride,
                                                        scale, input, partial_acc, partial_m,
                                                        partial_l, out, token_base, stream);
        return;
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        launch_partials<CacheInput, false, false, false, true>(q_heads, splits, kv_heads, q, q_stage,
                                                               positions, cache, tab_rows,
                                                               table_stride, scale, input,
                                                               partial_acc, partial_m, partial_l,
                                                               out, token_base, stream);
        return;
    case KvCacheStorage::Int8Group64:
        // Int8Group64 decode rides the KInt8 WMMA kernel (rotated-Q staging, G64
        // dequant of K/V): the KV bytes are halved vs bf16 and the WMMA QK/PV path
        // is bandwidth-driven, so this is the release-critical KV-compression path.
        launch_partials<CacheInput, false, false, false, false, true>(
            q_heads, splits, kv_heads, q, q_stage, positions, cache, tab_rows, table_stride, scale,
            input, partial_acc, partial_m, partial_l, out, token_base, stream);
        return;
    default:
        launch_partials<CacheInput, false, false>(q_heads, splits, kv_heads, q, q_stage, positions,
                                                  cache, tab_rows, table_stride, scale, input,
                                                  partial_acc, partial_m, partial_l, out, token_base,
                                                  stream);
        return;
    }
}

} // namespace

void causal_decode_t1_wmma_cached_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache,
                                         CausalAttentionExecutionEnvelope envelope,
                                         WorkspaceArena& workspace, Tensor& out,
                                         cudaStream_t stream) {
    const std::int32_t q_heads = q.ne[1];
    std::int32_t splits       = causal_attention_split_capacity(q_heads, 1, cache.storage, envelope);
    // B22 per-warp splits (Int8Group64): the KInt8 kernel maps 4 warps per CTA to 4
    // different splits, so the split count is the parallelism dial. Measured @32k:
    // 69 splits = 21.3 t/s, 276 = 23.2 (saturated), 552 = 23.3. Default x4 (the
    // engine tier count was tuned for one-warp-per-CTA); NINFER_WMMA_SPLITS still
    // overrides when set.
    if (cache.storage == KvCacheStorage::Int8Group64) { splits *= 4; }
    {
        static const int wmma_splits_override = [] {
            const char* e = std::getenv("NINFER_WMMA_SPLITS");
            return e != nullptr ? std::atoi(e) : 0;
        }();
        if (wmma_splits_override > 0) { splits = wmma_splits_override; }
    }
    Tensor partial_acc         = allocate_acc(workspace, q_heads, splits);
    Tensor partial_m           = allocate_stat(workspace, q_heads, splits);
    Tensor partial_l           = allocate_stat(workspace, q_heads, splits);
    const PagedKVBatchLayerView batch_cache = single_row_paged_kv_batch_view(cache);
    const std::int32_t kv_heads    = cache.num_kv_heads;
    const std::int32_t table_stride = batch_cache.block_tables.ne[1];
    // B22: per-layer Q-staging scratch (rotated+G64-quantized Q codes + scales),
    // written once by causal_decode_t1_q_stage_kernel and read by every split CTA
    // (was: redundantly recomputed inside each split CTA).
    Tensor q_stage = workspace.alloc(
        DType::U8, {kv_heads * kDecodeWmmaWarpRows *
                    (kDecodeWmmaD + kKVCacheInt8Groups * static_cast<std::int32_t>(sizeof(__half)))});
    // AMD DEBUG (MTP attention health): the cached T=1 route is chosen when the queried row is
    // already in the cache. A zero result here but a healthy one from the SIMT route means the
    // split/grid derivation, not the data, is dropping every key.
    static int t1wmma_dbg_calls = 0;
    if (std::getenv("NINFER_T1WMMA_DBG") != nullptr && t1wmma_dbg_calls++ < 8) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::int32_t host_pos = -1;
        CUDA_CHECK(cudaMemcpy(&host_pos, positions.data, sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        std::int32_t tabs[2] = {-1, -1};
        if (batch_cache.block_tables.data != nullptr) {
            CUDA_CHECK(cudaMemcpy(tabs, batch_cache.block_tables.data, 2 * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost));
        }
        std::fprintf(stderr,
                     "[t1wmma-dbg] q_ne=[%d,%d,%d] qh=%d kv=%d storage=%d splits=%d pos=%d "
                     "env=[%u,%u] tab_stride=%d tab=[%d,%d] k_scale=%d v_scale=%d\n",
                     q.ne[0], q.ne[1], q.ne[2], q_heads, kv_heads, static_cast<int>(cache.storage),
                     splits, host_pos, envelope.min_visible_keys, envelope.max_visible_keys,
                     table_stride, tabs[0], tabs[1],
                     cache.k_scale_pages.data != nullptr ? 1 : 0,
                     cache.v_scale_pages.data != nullptr ? 1 : 0);
        // Are the int4 dequant factors finite? A zero/inf scale turns every QK product into
        // NaN, which the reducer then reads as "no keys".
        if (cache.k_scale_pages.data != nullptr) {
            __half probe[8];
            CUDA_CHECK(cudaMemcpy(probe, cache.k_scale_pages.data, sizeof(probe),
                                  cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[t1wmma-dbg] k_scale[0..7]=");
            for (auto h : probe) { std::fprintf(stderr, " %.6g", static_cast<float>(h)); }
            std::fprintf(stderr, "\n");
        }
        {
            // q itself: a NaN here is upstream of the kernel, not a cache problem.
            const std::size_t qn_elems = static_cast<std::size_t>(q.ne[0]) * q.ne[1] * q.ne[2];
            std::vector<__nv_bfloat16> qh(qn_elems);
            CUDA_CHECK(cudaMemcpy(qh.data(), q.data, qn_elems * 2, cudaMemcpyDeviceToHost));
            int bad = 0;
            float mx = 0.0f;
            for (auto h : qh) {
                float f = __bfloat162float(h);
                if (!std::isfinite(f)) { ++bad; }
                if (f > mx) { mx = f; }
            }
            std::fprintf(stderr, "[t1wmma-dbg] q(n=%zu) nonfinite=%d max=%.6g k_pages.ne=[%d,%d,%d,%d]\n",
                         qn_elems, bad, mx, cache.k_pages.ne[0], cache.k_pages.ne[1],
                         cache.k_pages.ne[2], cache.k_pages.ne[3]);
        }
        std::fflush(stderr);
    }
    launch_partials_for_storage<CausalCachedInput>(
        cache.storage, q_heads, splits, kv_heads, q, q_stage, positions, batch_cache, nullptr,
        table_stride, scale, CausalCachedInput{}, partial_acc, partial_m, partial_l, out, 0,
        stream);
    if (std::getenv("NINFER_T1WMMA_DBG2") != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t n_acc = static_cast<std::size_t>(kCausalDecodeHeadDim) * q_heads * splits;
        const std::size_t n_st  = static_cast<std::size_t>(q_heads) * splits;
        std::vector<float> acc(n_acc), mm(n_st), ll(n_st);
        CUDA_CHECK(cudaMemcpy(acc.data(), partial_acc.data, n_acc * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mm.data(), partial_m.data, n_st * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ll.data(), partial_l.data, n_st * 4, cudaMemcpyDeviceToHost));
        auto summarize = [](const std::vector<float>& v, const char* name) {
            double sum = 0.0;
            float mx = -3.4e38f;
            int nonzero = 0;
            for (float f : v) {
                if (f != 0.0f) { ++nonzero; }
                if (f > mx) { mx = f; }
                sum += std::fabs(static_cast<double>(f));
            }
            std::fprintf(stderr, " %s(n=%zu nonzero=%d max=%.6g absum=%.6g)", name, v.size(), nonzero,
                         mx, sum);
        };
        std::fprintf(stderr, "[t1wmma-dbg2] splits=%d", splits);
        summarize(acc, "acc");
        summarize(mm, "m");
        summarize(ll, "l");
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
    (void)envelope;
}

// Append variant: the main decode path (causal_softmax_attention with fresh K/V) must
// stage the new token's K/V into the paged cache, then attend. The kernel does both
// under CacheInput::writes_cache.
void causal_decode_t1_wmma_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope,
                                  WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t q_heads = q.ne[1];
    static const bool simt_on = [] {
        const char* e = std::getenv("NINFER_ATTN_SIMT");
        return e != nullptr && e[0] == '1';
    }();
    if (simt_on) {
        std::fprintf(stderr, "[simt-launch] entry storage=%d qh=%d pos_ne0=%lld splits? env=%s\n",
                     (int)cache.storage, q_heads, (long long)positions.ne[0],
                     std::getenv("NINFER_ATTN_SIMT") ? std::getenv("NINFER_ATTN_SIMT") : "?");
        std::fflush(stderr);
    }
    std::int32_t splits       = causal_attention_split_capacity(q_heads, 1, cache.storage, envelope);
    // B22 per-warp splits (Int8Group64): the KInt8 kernel maps 4 warps per CTA to 4
    // different splits, so the split count is the parallelism dial. Measured @32k:
    // 69 splits = 21.3 t/s, 276 = 23.2 (saturated), 552 = 23.3. Default x4 (the
    // engine tier count was tuned for one-warp-per-CTA); NINFER_WMMA_SPLITS still
    // overrides when set.
    if (cache.storage == KvCacheStorage::Int8Group64) { splits *= 4; }
    {
        static const int wmma_splits_override = [] {
            const char* e = std::getenv("NINFER_WMMA_SPLITS");
            return e != nullptr ? std::atoi(e) : 0;
        }();
        if (wmma_splits_override > 0) { splits = wmma_splits_override; }
    }
    Tensor partial_acc         = allocate_acc(workspace, q_heads, splits);
    Tensor partial_m           = allocate_stat(workspace, q_heads, splits);
    Tensor partial_l           = allocate_stat(workspace, q_heads, splits);
    const std::int32_t kv_heads = cache.num_kv_heads;
    // B22: per-layer Q-staging scratch (rotated+G64-quantized Q codes + scales),
    // written once by causal_decode_t1_q_stage_kernel and read by every split CTA
    // (was: redundantly recomputed inside each split CTA).
    Tensor q_stage = workspace.alloc(
        DType::U8, {kv_heads * kDecodeWmmaWarpRows *
                    (kDecodeWmmaD + kKVCacheInt8Groups * static_cast<std::int32_t>(sizeof(__half)))});
    const std::int32_t* tab_rows =
        table_rows.data != nullptr ? static_cast<const std::int32_t*>(table_rows.data) : nullptr;
    const std::int32_t table_stride = cache.block_tables.ne[1];
    CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                            static_cast<const __nv_bfloat16*>(v.data)};
    // TEMP: per-call attention kernel GPU time (NINFER_ATTN_TIME=1).
    static const bool atn_time = [] {
        const char* e = std::getenv("NINFER_ATTN_TIME");
        return e != nullptr && e[0] == '1';
    }();
    if (atn_time) {
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, stream);
        launch_partials_for_storage<CausalAppendInput>(
            cache.storage, q_heads, splits, kv_heads, q, q_stage, positions, cache, tab_rows,
            table_stride, scale, input, partial_acc, partial_m, partial_l, out, 0, stream);
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        static int cnt = 0;
        if (++cnt <= 24) {
            std::fprintf(stderr, "[attn-kernel] call=%d window=%d kernel+reduce=%.3f ms\n", cnt,
                         positions.ne[0], ms);
            std::fflush(stderr);
        }
    } else {
        launch_partials_for_storage<CausalAppendInput>(
            cache.storage, q_heads, splits, kv_heads, q, q_stage, positions, cache, tab_rows,
            table_stride, scale, input, partial_acc, partial_m, partial_l, out, 0, stream);
    }
    if (std::getenv("NINFER_CACHE_APPEND_DUMP") != nullptr) {
        // Dump the just-appended K row (pos q = positions[0]) from cache to verify
        // decode's append wrote the correct K (== attn_kn_dec for that step).
        const std::int32_t qpos = positions.ne[0] > 0
                                      ? static_cast<std::int32_t>(
                                            *static_cast<const std::int32_t*>(positions.data))
                                      : 0;
        static const int kCacheDumpCap = 6;
        static int cache_append_step = 0;
        if (cache_append_step < kCacheDumpCap) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::int32_t physical_page = static_cast<const std::int32_t*>(
                cache.block_tables.data)[qpos >> kPagedKVPageShift];
            // host-side cache offset: [page][head][token][d], strides d=2B, token=256,
            // head=256*64, page=256*64*num_kv_heads.
            const std::int64_t off =
                static_cast<std::int64_t>(physical_page) * kCausalDecodeHeadDim * 64 *
                    cache.num_kv_heads +
                static_cast<std::int64_t>(qpos & kPagedKVPageMask) * kCausalDecodeHeadDim;
            std::vector<std::uint16_t> kb(256);
            CUDA_CHECK(cudaMemcpy(kb.data(),
                                  static_cast<const std::uint8_t*>(cache.k_pages.data) + off * 2,
                                  256 * 2, cudaMemcpyDeviceToHost));
            char fn[128];
            std::snprintf(fn, sizeof(fn), "D:/ninfer-hip-migrate/build/cache_app_k_%d.bin",
                          cache_append_step);
            FILE* kf = std::fopen(fn, "wb");
            if (kf) { std::fwrite(kb.data(), 2, 256, kf); std::fclose(kf); }
            std::fprintf(stderr, "[cache-app] step=%d pos=%d page=%d off=%lld h0k0=%u,%u,%u,%u\n",
                         cache_append_step, qpos, physical_page, (long long)off,
                         static_cast<unsigned>(kb[0]), static_cast<unsigned>(kb[1]),
                         static_cast<unsigned>(kb[2]), static_cast<unsigned>(kb[3]));
            std::fflush(stderr);
            // AMD debug: also dump the appended V int8 codes + G64 scales for the same
            // token/head, so a host replay can dequantize and compare to the bf16
            // input V (attn_v_dec_s0.bin). V plane layout mirrors K (256 bytes/row);
            // scale plane is [page][head][token][group] with 4 groups (2B each).
            if (cache.v_pages.data != nullptr) {
                std::vector<std::uint8_t> vb(256);
                CUDA_CHECK(cudaMemcpy(vb.data(),
                                      static_cast<const std::uint8_t*>(cache.v_pages.data) + off,
                                      256, cudaMemcpyDeviceToHost));
                std::snprintf(fn, sizeof(fn), "D:/ninfer-hip-migrate/build/cache_app_v_%d.bin",
                              cache_append_step);
                FILE* vf = std::fopen(fn, "wb");
                if (vf) { std::fwrite(vb.data(), 1, 256, vf); std::fclose(vf); }
                if (cache.v_scale_pages.data != nullptr) {
                    const std::int64_t soff =
                        static_cast<std::int64_t>(physical_page) * 4 * 64 * cache.num_kv_heads +
                        static_cast<std::int64_t>(qpos & kPagedKVPageMask) * 4;
                    std::vector<std::uint16_t> vsb(4);
                    CUDA_CHECK(cudaMemcpy(vsb.data(),
                                          static_cast<const std::uint8_t*>(
                                              cache.v_scale_pages.data) +
                                              soff * 2,
                                          4 * 2, cudaMemcpyDeviceToHost));
                    std::snprintf(fn, sizeof(fn),
                                  "D:/ninfer-hip-migrate/build/cache_app_vscale_%d.bin",
                                  cache_append_step);
                    FILE* sf = std::fopen(fn, "wb");
                    if (sf) { std::fwrite(vsb.data(), 2, 4, sf); std::fclose(sf); }
                    std::fprintf(stderr,
                                 "[cache-app] V dump step=%d v0..3=%d,%d,%d,%d vscale=%u,%u,%u,%u\n",
                                 cache_append_step, (int)vb[0], (int)vb[1], (int)vb[2],
                                 (int)vb[3], static_cast<unsigned>(vsb[0]),
                                 static_cast<unsigned>(vsb[1]), static_cast<unsigned>(vsb[2]),
                                 static_cast<unsigned>(vsb[3]));
                    std::fflush(stderr);
                }
            }
            ++cache_append_step;
        }
    }
    (void)envelope;
}

// AMD port (gfx1201): multi-token decode-verify attention.
//
// Why: the speculative-verify pass feeds 2..8 tokens from ONE sequence through the
// attention op. The small-T tiled kernel is the natural multi-token route, but on
// gfx1201 it measures 78.2 ms of a 119.2 ms verify round at a 5.6k window (4 tokens,
// 16 full-attention layers = ~185 MB of KV, i.e. ~270x the bandwidth floor), while the
// tuned T=1 SIMT kernel covers the same window in ~0.1 ms/layer because its grid is
// q_heads/2 x splits and it is latency-, not bandwidth-bound.
//
// So: run the T=1 SIMT kernel once per verify token. Each call appends that token's
// fresh K/V into the paged cache and attends up to its own position (token_base selects
// the column of q / k / v / positions), which is exactly the causal mask the verify
// batch needs, and the launcher is in-stream so token t sees tokens < t already written.
// Partials are reused across the loop; the reducer is re-run per column (1-token window,
// Offset=true) so the output layout stays [token][q_head][d].
bool causal_decode_t1_simt_multi_token_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                             const Tensor& positions, const Tensor& table_rows,
                                             float scale, PagedKVBatchLayerView cache,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t tokens, WorkspaceArena& workspace,
                                             Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "causal_decode_t1_simt_multi_token_launch";
    const std::int32_t q_heads = q.ne[1];
    if (tokens < 2 || tokens > 8) { return false; }
    if (q_heads != CausalD256H24Kv4::QHeads && q_heads != CausalD256H16Kv2::QHeads) { return false; }
    // Only the storages the SIMT decoder understands; Int8Group64 needs the KInt8 WMMA
    // decoder, which this route does not carry (it stays on the small-T kernel).
    if (cache.storage != KvCacheStorage::BFloat16 && cache.storage != KvCacheStorage::Fp16 &&
        cache.storage != KvCacheStorage::Q4_0 &&
        cache.storage != KvCacheStorage::Fp8E4M3Row256 &&
        cache.storage != KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
        return false;
    }
    static const bool simt_off = [] {
        const char* e = std::getenv("NINFER_ATTN_SIMT");
        return e != nullptr && e[0] == '0';
    }();
    if (simt_off) { return false; }

    const std::int32_t kv_heads = cache.num_kv_heads;
    const std::int32_t splits =
        causal_attention_split_capacity(q_heads, 1, cache.storage, envelope);
    Tensor partial_acc = allocate_acc(workspace, q_heads, splits);
    Tensor partial_m   = allocate_stat(workspace, q_heads, splits);
    Tensor partial_l   = allocate_stat(workspace, q_heads, splits);
    Tensor q_stage     = workspace.alloc(
        DType::U8, {kv_heads * kDecodeWmmaWarpRows *
                    (kDecodeWmmaD + kKVCacheInt8Groups * static_cast<std::int32_t>(sizeof(__half)))});
    const std::int32_t* tab_rows =
        table_rows.data != nullptr ? static_cast<const std::int32_t*>(table_rows.data) : nullptr;
    const std::int32_t table_stride = cache.block_tables.ne[1];
    const CausalAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                                  static_cast<const __nv_bfloat16*>(v.data)};
    static int mt_path_calls = 0;
    if (std::getenv("NINFER_MTDBG") != nullptr && mt_path_calls++ < 12) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
        if (cap == cudaStreamCaptureStatusNone) {
            std::int32_t row0 = -1;
            if (tab_rows != nullptr) {
                CUDA_CHECK(cudaMemcpy(&row0, tab_rows, sizeof(std::int32_t),
                                      cudaMemcpyDeviceToHost));
            }
            std::fprintf(stderr,
                         "[mt-path] tokens=%d splits=%d storage=%d q_heads=%d kv_heads=%d "
                         "tab_stride=%d row0=%d table_rows=%s env=[%u,%u]\n",
                         tokens, splits, static_cast<int>(cache.storage), q_heads, kv_heads,
                         table_stride, row0, tab_rows != nullptr ? "yes" : "null",
                         static_cast<unsigned>(envelope.min_visible_keys),
                         static_cast<unsigned>(envelope.max_visible_keys));
            std::fflush(stderr);
        }
    }
    static int pdbg_calls = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        launch_partials_for_storage<CausalAppendInput>(
            cache.storage, q_heads, splits, kv_heads, q, q_stage, positions, cache, tab_rows,
            table_stride, scale, input, partial_acc, partial_m, partial_l, out, t, stream);
        // AMD DEBUG (MTP zero attention): the multi-token route returns an all-zero column on the
        // MTP cache while the same code is healthy on the target cache. Reading the partials back
        // separates "the split kernel wrote neutral slots" from "the reducer dropped them".
        if (t == 0 && tokens == 2 && std::getenv("NINFER_MTP_PDBG") != nullptr &&
            pdbg_calls < 10) {
            cudaStreamCaptureStatus pcap = cudaStreamCaptureStatusNone;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &pcap));
            if (pcap == cudaStreamCaptureStatusNone) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                ++pdbg_calls;
                float m0 = 0.0f, l0 = 0.0f;
                CUDA_CHECK(cudaMemcpy(&m0, partial_m.data, sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&l0, partial_l.data, sizeof(float), cudaMemcpyDeviceToHost));
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                CUDA_CHECK(cudaMemcpy(acc, partial_acc.data, 4 * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                __nv_bfloat16 o[4] = {};
                CUDA_CHECK(cudaMemcpy(o, out.data, 4 * sizeof(__nv_bfloat16),
                                      cudaMemcpyDeviceToHost));
                std::int32_t p0 = -999;
                CUDA_CHECK(cudaMemcpy(&p0, positions.data, sizeof(p0),
                                      cudaMemcpyDeviceToHost));
                std::fprintf(stderr,
                             "[mtp-pdbg] n=%d tokens=%d splits=%d pos0=%d m0=%.6g l0=%.6g "
                             "acc0=%.6g %.6g out=%g %g\n",
                             pdbg_calls, tokens, splits, p0, m0, l0, acc[0], acc[1],
                             __bfloat162float(o[0]), __bfloat162float(o[1]));
                std::fflush(stderr);
            }
        }
    }
    (void)op;
    return true;
}

} // namespace ninfer::ops::detail
