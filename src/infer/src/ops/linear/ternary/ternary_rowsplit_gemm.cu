// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_rowsplit_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/ternary/ternary_launch.h"
#include "ops/linear/ternary/ternary_dispatch.h" // AMD port: verify-mode kernel selection
#include "ops/linear/ternary/ternary_rowsplit_gemv.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh"
#include "ops/linear/ternary/ternary_rowsplit_mma_prefill.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <windows.h> // AMD port: QueryPerformanceCounter for NINFER_TERN_TRACE

namespace ninfer::ops::detail {
namespace {

// B22 GEMV split-K: lazily-grown fp32 partial buffer [rows x k_splits]. Written
// exactly once per (row, split) by the ksplit kernel (no atomics -> deterministic),
// summed by ternary_gemv_ksplit_reduce_kernel.
// B25: cudaMalloc during a HIP-graph capture is illegal (StreamCaptureUnsupported).
// Pre-warm() is called right before cudaStreamBeginCapture (decode_graph.cpp), so the
// allocation happens outside capture; any capture-time call then hits the cached
// buffer and never allocates. Fixed 16 MiB ceiling covers every decode shape
// (largest ksplit consumer: n=6144, ks=3 -> 73 KiB; 16 MiB is generous headroom).
float* ternary_ksplit_partial_buffer(std::size_t bytes) {
    static float* buf = nullptr;
    static std::size_t cap = 0;
    const std::size_t kMaxBytes = 16u << 20;
    if (buf == nullptr) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf), kMaxBytes));
        cap = kMaxBytes;
    }
    if (bytes > cap) {
        throw std::runtime_error("ternary ksplit partial buffer exceeded");
    }
    return buf;
}

// ---------------------------------------------------------------------------
// AMD port diagnostics: NINFER_TERN_TRACE=1 prints one line per (path, shape)
// with the wall-clock time accumulated by that path, so a full forward pass can
// be attributed to kernels without a profiler. Off by default (one getenv).
// ---------------------------------------------------------------------------
struct TernaryTrace {
    bool on = false;
    struct Entry {
        const char* path;
        std::int32_t n;
        std::int32_t k;
        std::int32_t t;
        double ms;
        long calls;
    };
    Entry entries[24] = {};
    int count         = 0;

    static TernaryTrace& instance() {
        static TernaryTrace t = [] {
            TernaryTrace x;
            const char* e = std::getenv("NINFER_TERN_TRACE");
            x.on          = e != nullptr && e[0] == '1';
            if (x.on) {
                std::atexit(&TernaryTrace::report);
            }
            return x;
        }();
        return t;
    }

    void add(const char* path, std::int32_t n, std::int32_t k, std::int32_t t, double ms) {
        if (!on) {
            return;
        }
        for (int i = 0; i < count; ++i) {
            if (entries[i].n == n && entries[i].k == k && entries[i].t == t) {
                entries[i].ms += ms;
                entries[i].calls += 1;
                return;
            }
        }
        if (count < 24) {
            entries[count++] = Entry{path, n, k, t, ms, 1};
        }
    }

    static void report() {
        TernaryTrace& t = instance();
        if (!t.on) {
            return;
        }
        std::fprintf(stderr, "[tern-trace] %d shapes\n", t.count);
        for (int i = 0; i < t.count; ++i) {
            std::fprintf(stderr, "[tern-trace] %-10s n=%6d k=%6d t=%4d calls=%5ld %8.2f ms\n",
                         t.entries[i].path, t.entries[i].n, t.entries[i].k, t.entries[i].t,
                         t.entries[i].calls, t.entries[i].ms);
        }
        std::fflush(stderr);
    }
};

struct TernaryTraceScope {
    const char* path;
    std::int32_t n, k, t;
    bool on;
    long long t0;
    explicit TernaryTraceScope(const char* p, std::int32_t nn, std::int32_t kk, std::int32_t tt)
        : path(p), n(nn), k(kk), t(tt), on(TernaryTrace::instance().on), t0(0) {
        if (on) {
            LARGE_INTEGER c;
            QueryPerformanceCounter(&c);
            t0 = c.QuadPart;
        }
    }
    ~TernaryTraceScope() {
        if (!on) {
            return;
        }
        LARGE_INTEGER c, f;
        QueryPerformanceCounter(&c);
        QueryPerformanceFrequency(&f);
        const double ms =
            static_cast<double>(c.QuadPart - t0) * 1000.0 / static_cast<double>(f.QuadPart);
        TernaryTrace::instance().add(path, n, k, t, ms);
    }
};

} // namespace
} // namespace ninfer::ops::detail

namespace ninfer::ops::detail {
namespace {

// Decode (T == 1) takes the warp-per-row GEMV for PQ2_0. K is a whole number of 128-groups for
// every width in this model, so that kernel needs no column guard.
void launch_pq2_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if ((w.k % 128) != 0 || x.ne[1] != 1) {
        throw std::invalid_argument("ternary gemv: expected one token and a whole-group K");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    ternary_pq2_gemv_kernel<<<grid, kGemvWarpsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        groups_per_row, static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
    CUDA_CHECK(cudaGetLastError());
}

// Small-token-tile GEMV: weights are read once for up to 4 tokens, which is what makes the
// speculative verify pass (T = draft + 1) cheap. Falls back to the reference tiled kernel beyond
// that, and for PTQ1_0 / padded-K weights.
void launch_pq2_gemv_tile(const Tensor& x, const Weight& w, Tensor& out,
                          std::int32_t out_row_stride, std::int32_t tokens,
                          cudaStream_t stream) {
    if ((w.k % 128) != 0) {
        throw std::invalid_argument("ternary gemv: K must be a whole number of 128-groups");
    }
    const std::int32_t groups_per_row = w.k / 128;
    const unsigned grid               = static_cast<unsigned>(div_up(w.n, kGemvWarpsPerBlock));
    const dim3 block(kGemvWarpsPerBlock * 32, 1u, 1u);
    if (tokens <= 1) {
        ternary_pq2_gemv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row, static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
    } else {
        ternary_pq2_gemv_tile_kernel<4><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
            groups_per_row, tokens, out_row_stride, static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Storage, class Atom, int kTileT>
void launch_gemm(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                 cudaStream_t stream) {
    const std::int32_t rows = w.n;
    const std::int32_t k    = w.k;
    const std::int32_t t    = x.ne[1];
    if (k % Storage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    if (out_row_stride < rows) {
        throw std::invalid_argument("ternary linear: output row stride is smaller than the tile");
    }
    const std::int32_t groups_per_row = k / Storage::kGroupK;

    const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(div_up(t, kTileT)), 1u);
    constexpr dim3 block(Storage::kGroupK, 1u, 1u);

    ternary_rowsplit_gemm_kernel<Storage, Atom, kTileT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), rows, k, t, groups_per_row, out_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <int kTileT>
void launch_by_qtype(const Tensor& x, const Weight& w, Tensor& out, std::int32_t out_row_stride,
                     cudaStream_t stream) {
    switch (w.qtype) {
    case QType::PTQ1_0_G128:
        launch_gemm<PTQ1RowSplitStorage, PTQ1SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                    stream);
        return;
    case QType::PQ2_0_G128:
        launch_gemm<PQ2RowSplitStorage, PQ2SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                  stream);
        return;
    case QType::T2_G128_FP16:  // iamwavecut v3: same row layout, bit0/bit1 codec
        launch_gemm<T2RowSplitStorage, T2SimtDecodeAtom, kTileT>(x, w, out, out_row_stride,
                                                                stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("ternary linear: unsupported weight qtype");
}

} // namespace

// A PQ2_0 weight can take the GEMV family whenever the layout did not pad K past the real width --
// true for every width in this model (5120/6144/10240/17408 are all whole 128-groups), and checked
// here rather than assumed, because the GEMV reads whole groups without a column guard.
bool gemv_admits(const Tensor& x, const Weight& w, std::int32_t max_tokens) {
    return (w.qtype == QType::PQ2_0_G128 || w.qtype == QType::T2_G128_FP16) &&
           w.qhigh == nullptr && w.padded_shape[1] == w.k && (w.k % 128) == 0 &&
           x.ne[1] >= 1 && x.ne[1] <= max_tokens;
}

void launch_ternary_gemm_t1(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream) {
    if (gemv_admits(x, w, 1)) {
        // AMD port (gfx1201): the shared-memory-activation GEMV collapses the
        // per-row activation re-read that capped the one-row-per-warp kernel at
        // ~45% of DRAM bandwidth; measured 1.4-1.7x on the decode shapes
        // (probe/bench_decode_gemv.cu). Rows must be even (all model widths are).
        // NINFER_GEMV_WMMA=1: T=1 through the bf16 WMMA path (pq2_rowsplit_wmma2),
        // the NInfer-style tensor-core GEMV on RDNA4 - A/B against the SIMT v4.
        static const bool gemv_wmma = [] {
            const char* e = std::getenv("NINFER_GEMV_WMMA");
            return e != nullptr && e[0] == '1';
        }();
        if (gemv_wmma) {
            TernaryTraceScope trace("gemv_wmma1", w.n, w.k, x.ne[1]);
            const int t2 = (w.qtype == QType::T2_G128_FP16) ? 1 : 0;
            const dim3 wgrid((w.n + 31) / 32, 1);
            pq2_rowsplit_wmma2_r32c16_kernel<<<wgrid, kPQ2Wmma2R32C16Threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales),
                static_cast<__nv_bfloat16*>(out.data), w.n, w.k, 1, w.n, t2);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if ((w.n % 2) == 0 && (w.k % 8) == 0) {
            // MEASURED (engine, real task, 200 generated tokens): forcing 1 row per warp
            // (8 rows/CTA) on the narrow shapes made decode 43.9 -> 42.8 t/s, and 2 rows
            // per warp is what the wide shapes want anyway. The 16-rows-per-CTA form is
            // the right one everywhere; a rows-per-warp template is kept in
            // ternary_rowsplit_gemv.cuh (ternary_pq2_gemv_smem_rows_kernel) for sweeps.
            TernaryTraceScope trace("gemv_smem", w.n, w.k, x.ne[1]);
            const std::int32_t groups = w.k / 128;
            const unsigned grid = static_cast<unsigned>((w.n / 2 + 7) / 8);
            static const bool gpu_time = [] {
                const char* e = std::getenv("NINFER_TERN_GPU_TIME");
                return e != nullptr && e[0] == '1';
            }();
            // Default is v4 (four groups/iter) everywhere -- the verified engine winner.
            // v16 wins standalone (612 GB/s on gate_up) but LOSES in the engine
            // (register pressure drops occupancy; @1024 31.95 vs v4 33.67),
            // so it stays opt-in via NINFER_GEMV_V16=1 (probe-only result).
            // NINFER_GEMV_V4=0 forces single; NINFER_GEMV_DUAL=1 the two-group variant.
            static const bool gemv_v16 = [] {
                const char* e = std::getenv("NINFER_GEMV_V16");
                return e != nullptr && e[0] == '1';  // default OFF
            }();
            static const bool gemv_v4 = [] {
                const char* e = std::getenv("NINFER_GEMV_V4");
                return e == nullptr || e[0] != '0';  // default ON
            }();
            static const bool gemv_dual = [] {
                const char* e = std::getenv("NINFER_GEMV_DUAL");
                return e != nullptr && e[0] == '1';
            }();
            static const bool gemv_v4x4 = [] {
                const char* e = std::getenv("NINFER_GEMV_V4X4");
                return e != nullptr && e[0] == '1';  // default OFF
            }();
            static const bool gemv_v5 = [] {
                const char* e = std::getenv("NINFER_GEMV_V5");
                return e != nullptr && e[0] == '1';  // default OFF
            }();
            static const bool gemv_v8 = [] {
                const char* e = std::getenv("NINFER_GEMV_V8");
                return e != nullptr && e[0] == '1';  // default OFF
            }();
            static const bool gemv_v9 = [] {
                const char* e = std::getenv("NINFER_GEMV_V9");
                return e == nullptr || e[0] != '0';  // default ON (v9 beats v4 everywhere)
            }();
            const bool use_v16 = gemv_v16 && gemv_v4 && groups <= 64;
            const auto launch_one = [&](const __nv_bfloat16* xd, const std::uint8_t* qd,
                                        const std::uint8_t* sd, __nv_bfloat16* od,
                                        std::int32_t n, std::int32_t g) {
                if (w.qtype == QType::T2_G128_FP16) {
                    if (use_v16) {
                        ternary_pq2_gemv_smem_v16_kernel<true><<<grid, kGemvSmemThreads, w.k * 2,
                                                                 stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v9) {
                        // B22: split-K for the small-n decode shapes -- plain v9
                        // launches (n/16) CTAs (320 for n=5120 = 15 warps/SM = 23%
                        // occupancy). NINFER_GEMV_KSPLIT=0 disables.
                        static const bool gemv_ksplit = [] {
                            const char* e = std::getenv("NINFER_GEMV_KSPLIT");
                            return e == nullptr || e[0] != '0';
                        }();
                        // B22 bench (bench_gemv_wall.cu, real n=5120/k=17408 shape):
                        // ks=1 -> 433 GB/s, ks=2 -> 580, ks=4 -> 594 (best),
                        // ks=7 -> 577, ks=14 -> 517. Target ~1280 CTAs (20480/n).
                        // B25: ksplit's partial buffer used to cudaMalloc lazily at
                        // launch time, which is illegal while a HIP graph is being
                        // captured (hipErrorStreamCaptureUnsupported) and crashed the
                        // decode. The buffer is now a fixed 16 MiB pre-allocation; it
                        // is allocated on the first GEMV (prefill runs before decode
                        // graph capture, so it is always warm by then) and never
                        // reallocated, making ksplit graph-safe.
                        const std::int32_t ks_raw = (20480 + n - 1) / n;
                        const std::int32_t ks =
                            (gemv_ksplit && ks_raw < 8) ? ks_raw : 1;
                        const std::int32_t g_base =
                            (ks > 1) ? ((((g + ks - 1) / ks) + 3) & ~3) : g;
                        if (ks > 1 && g_base >= 4 && (n % 2) == 0) {
                            float* partial = ternary_ksplit_partial_buffer(
                                static_cast<std::size_t>(n) * ks * sizeof(float));
                            const dim3 g2(grid, static_cast<unsigned>(ks), 1u);
                            ternary_pq2_gemv_smem_v9_ksplit_kernel<true>
                                <<<g2, kGemvSmemThreads,
                                   g_base * 128 *
                                       static_cast<std::int32_t>(sizeof(__nv_bfloat16)),
                                   stream>>>(xd, qd, sd, partial, n, g, g_base, ks);
                            ternary_gemv_ksplit_reduce_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
                                partial, od, n, ks);
                        } else {
                            ternary_pq2_gemv_smem_v9_kernel<true><<<grid, kGemvSmemThreads, w.k * 2,
                                                                    stream>>>(xd, qd, sd, od, n, g);
                        }
                    } else if (gemv_v8) {
                        ternary_pq2_gemv_smem_v8_kernel<true, 4><<<grid, kGemvSmemThreads, w.k * 2,
                                                                  stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v5) {
                        ternary_pq2_gemv_smem_v5_kernel<true><<<grid, kGemvSmemThreads, w.k * 2,
                                                                stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v4x4 && g >= 4) {
                        ternary_pq2_gemv_smem_v4x4_kernel<true><<<grid, kGemvSmemThreads, w.k * 2,
                                                                  stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v4) {
                        ternary_pq2_gemv_smem_v4_kernel<true><<<grid, kGemvSmemThreads, w.k * 2,
                                                                stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_dual) {
                        ternary_pq2_gemv_smem_dual_kernel<true><<<grid, kGemvSmemThreads, w.k * 2,
                                                                  stream>>>(xd, qd, sd, od, n, g);
                    } else {
                        ternary_pq2_gemv_smem_kernel<true><<<grid, kGemvSmemThreads, w.k * 2,
                                                             stream>>>(xd, qd, sd, od, n, g);
                    }
                } else {
                    if (use_v16) {
                        ternary_pq2_gemv_smem_v16_kernel<false><<<grid, kGemvSmemThreads, w.k * 2,
                                                                  stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v9) {
                        ternary_pq2_gemv_smem_v9_kernel<false><<<grid, kGemvSmemThreads, w.k * 2,
                                                                  stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v8) {
                        ternary_pq2_gemv_smem_v8_kernel<false, 4><<<grid, kGemvSmemThreads, w.k * 2,
                                                                    stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v5) {
                        ternary_pq2_gemv_smem_v5_kernel<false><<<grid, kGemvSmemThreads, w.k * 2,
                                                                  stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v4x4 && g >= 4) {
                        ternary_pq2_gemv_smem_v4x4_kernel<false><<<grid, kGemvSmemThreads, w.k * 2,
                                                                    stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_v4) {
                        ternary_pq2_gemv_smem_v4_kernel<false><<<grid, kGemvSmemThreads, w.k * 2,
                                                                 stream>>>(xd, qd, sd, od, n, g);
                    } else if (gemv_dual) {
                        ternary_pq2_gemv_smem_dual_kernel<false><<<grid, kGemvSmemThreads, w.k * 2,
                                                                   stream>>>(xd, qd, sd, od, n, g);
                    } else {
                        ternary_pq2_gemv_smem_kernel<false><<<grid, kGemvSmemThreads, w.k * 2,
                                                              stream>>>(xd, qd, sd, od, n, g);
                    }
                }
            };
            if (gpu_time) {
                cudaStreamSynchronize(stream);
                cudaEvent_t e0 = nullptr, e1 = nullptr;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, stream);
                launch_one(static_cast<const __nv_bfloat16*>(x.data),
                           static_cast<const std::uint8_t*>(w.qdata),
                           static_cast<const std::uint8_t*>(w.scales),
                           static_cast<__nv_bfloat16*>(out.data) /* token 0 row-major = out */,
                           w.n, groups);
                cudaEventRecord(e1, stream);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                static int gc = 0;
                if (x.ne[1] == 1 && ++gc <= 200) {
                    std::fprintf(stderr, "[gemv-t1] n=%d k=%d %.4f ms\n", w.n, w.k, ms);
                    std::fflush(stderr);
                }
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
            } else {
                launch_one(static_cast<const __nv_bfloat16*>(x.data),
                           static_cast<const std::uint8_t*>(w.qdata),
                           static_cast<const std::uint8_t*>(w.scales),
                           static_cast<__nv_bfloat16*>(out.data) /* token 0 row-major = out */,
                           w.n, groups);
            }
            const cudaError_t gemv_err = cudaGetLastError();
            if (gemv_err != cudaSuccess) {
                std::fprintf(stderr, "[smem-gemv-fail] n=%d k=%d err=%s\n", w.n, w.k,
                             cudaGetErrorString(gemv_err));
                std::fflush(stderr);
            }
            CUDA_CHECK(gemv_err);
            return;
        }
        launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    TernaryTraceScope trace("by_qtype1", w.n, w.k, x.ne[1]);
    launch_by_qtype<1>(x, w, out, out_row_stride, stream);
}

// ---------------------------------------------------------------------------
// Ternary PQ2_0 tensor-core path for the small-T regime (T = 2..8), implemented in
// ternary_rowsplit_mma_small_t.cuh. This is the batch fast path.
//
// The SIMT tile GEMV above reuses each weight across its tile but still pays one FMA per weight
// PER TOKEN, so its cost grows with T while its weight traffic does not: measured on a 322 MB
// PQ2_0 payload it takes 0.706 ms at T = 1 and 1.533 ms at T = 3, i.e. 2.2x the time for
// byte-identical traffic (211 GiB/s against 483 GiB/s). One mma.m16n8k16 carries 16 rows x 8
// tokens x 16 k, so every token in the tile rides the same weight read at no extra instruction
// cost.
//
// T >= 2 only: the T = 1 endpoint of that same measurement (0.603 against 0.706 ms) did NOT
// transfer to the real engine -- see the note on launch_ternary_gemm_t1 -- so decode stays on
// the GEMV. Selected inside launch_ternary_gemm_t8; roll back with NINFER_TERNARY_MMA=0.
// ---------------------------------------------------------------------------
bool mma_admits(const Weight& w, std::int32_t out_row_stride) {
    return (w.qtype == QType::PQ2_0_G128 || w.qtype == QType::T2_G128_FP16) &&
           w.qhigh == nullptr && w.padded_shape[1] == w.k &&
           (w.k % kTernaryMmaChunkK) == 0 && out_row_stride >= w.n;
}

void launch_pq2_mma_small_t(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, std::int32_t tokens,
                            cudaStream_t stream) {
    // The staging is static shared memory (a compile-time 32 KB), so there is no dynamic size to
    // pass and no cudaFuncSetAttribute opt-in to make -- which is also what keeps it legal inside
    // the captured CUDA graphs this op runs in.
    const unsigned grid = static_cast<unsigned>(div_up(w.n, kTernaryMmaRowsPerCta));
    ternary_pq2_mma_small_t_kernel<<<grid, kTernaryMmaThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), w.n,
        w.k, tokens, out_row_stride,
        static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
    CUDA_CHECK(cudaGetLastError());
}

// PREFILL BATCHING launcher: BlockRows(64) x BlockCols(32) x 128-K blocks.
// Three kernels share the validated fragment layouts (probe/prefill_probe.cu):
//   * pq2_rowsplit_wmma2_kernel -- register-decoded HW fragments, zero staging,
//     zero barriers, post-mma group scale (default; NINFER_PREFILL_SW=1 falls
//     back to the software-fragment kernel, NINFER_PREFILL_WMMA1=1 to the
//     smem-fragment WMMA kernel).
//   * pq2_rowsplit_wmma_kernel -- smem-fragment WMMA (v1).
//   * pq2_rowsplit_gemm_mma_kernel -- upstream-style ldmatrix + software mma.
void launch_pq2_mma_prefill(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    static const bool use_sw = [] {
        const char* v = std::getenv("NINFER_PREFILL_SW");
        return v != nullptr && std::string(v) == "1";
    }();
    static const bool use_wmma1 = [] {
        const char* v = std::getenv("NINFER_PREFILL_WMMA1");
        return v != nullptr && std::string(v) == "1";
    }();
    // MEASURED (probe/bench_small_t.cu, gfx1201):
    //   gate_up N=34816 K=5120, 47.3 MB of weights
    //     BN32 @T<=4 0.213 ms   BN16 @T<=4 0.198 ms
    //   down    N=5120  K=17408, 23.7 MB
    //     BN32 @T=4  0.171 ms   BN16 @T=4  0.134 ms   (1.28x)
    // The verify pass (T = 2..16) never fills a 32-token block, so a 32-wide token
    // tile makes it pay for tokens that do not exist. Prefill (T >= 32) keeps BN32,
    // where the wider block is what amortises the A-operand decode.
    // ON only for the speculative VERIFY pass (set via set_ternary_verify_mode). Prefill
    // keeps the WMMA path so its numerics are unchanged. MEASURED (probe/bench_small_t.cu,
    // gfx1201, gate_up N=34816 K=5120 = 47.3 MB):
    //   T=4  mma 16-token block 5.71 ms | wmma2 BN32 0.213 | multi 0.187 ms
    //   down N=5120 K=17408:                 wmma2 BN32 0.171 | multi 0.072 ms
    // Correctness: probe checks the multi kernel BIT-EXACT against running the T=1 smem
    // GEMV once per token, for every (N, K, T, stride) tried. It is a different accumulator
    // order from WMMA, so verify (whose output only goes through the acceptance tolerance)
    // may take it while prefill may not.
    static const bool gemv_multi = [] {
        const char* v = std::getenv("NINFER_TERN_GEMV_MULTI");
        return v == nullptr || v[0] != '0'; // NINFER_TERN_GEMV_MULTI=0 rolls back
    }();
    if (gemv_multi && ternary_verify_mode() && gemv_admits(x, w, 8) && (w.n % 16) == 0 &&
        (w.k % 8) == 0 && x.ne[1] >= 2 && x.ne[1] <= 8) {
        if (launch_pq2_gemv_smem_multi(static_cast<const __nv_bfloat16*>(x.data),
                                       static_cast<const std::uint8_t*>(w.qdata),
                                       static_cast<const std::uint8_t*>(w.scales),
                                       static_cast<__nv_bfloat16*>(out.data), w.n,
                                       w.k / 128, x.ne[1], out_row_stride, stream,
                                       static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16))) {
            return;
        }
    }
    static const bool no_bn16 = [] {
        const char* v = std::getenv("NINFER_PREFILL_BN32");
        return v != nullptr && v[0] == '1';
    }();
    // OPT-IN ONLY. The BN16 kernel matches the BN32 kernel bit-exactly in
    // probe/bench_small_t.cu for every shape and stride tried (N=1024/4096/5120/34816,
    // K=5120/17408, T=4..16, padded output strides included), but wiring it into the
    // engine CORRUPTS the forward pass: a 13-token prompt comes back as gibberish with
    // the engine's own (n=4096, k=5120, tokens=6, stride=10240) call, which the same
    // probe reproduces as a MATCH. Until that discrepancy is explained the engine keeps
    // the BN32 schedule; NINFER_PREFILL_BN16=1 enables the narrow block for experiments.
    static const bool bn16_opt_in = [] {
        const char* v = std::getenv("NINFER_PREFILL_BN16");
        return v != nullptr && v[0] == '1';
    }();
    if (!use_sw && !use_wmma1 && bn16_opt_in && !no_bn16 && tokens <= 16) {
        {
            static int printed = 0;
            if (printed < 40) {
                ++printed;
                std::fprintf(stderr,
                             "[prefill-bn16] n=%d k=%d tokens=%d stride=%d padk=%d qtype=%d xne=%lld,%lld,%lld,%lld xnb1=%lld outne1=%lld\n",
                             w.n, w.k, tokens, out_row_stride,
                             static_cast<int>(w.padded_shape[1]), static_cast<int>(w.qtype),
                             static_cast<long long>(x.ne[0]), static_cast<long long>(x.ne[1]),
                             static_cast<long long>(x.ne[2]), static_cast<long long>(x.ne[3]),
                             static_cast<long long>(x.nb[1]), static_cast<long long>(out.ne[1]));
                std::fflush(stderr);
            }
        }
        const dim3 grid16(static_cast<unsigned>(div_up(w.n, 64)),
                          static_cast<unsigned>(div_up(tokens, 16)), 1u);
        pq2_rowsplit_wmma2_bn16_kernel<<<grid16, PQ2RowSplitWmma2Schedule16::kThreads>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, w.k, tokens, out_row_stride, static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const unsigned grid_x = static_cast<unsigned>(div_up(w.n, 64));
    const unsigned grid_y = static_cast<unsigned>(div_up(tokens, 32));
    const dim3 grid(grid_x, grid_y, 1u);
    if (use_sw) {
        pq2_rowsplit_gemm_mma_kernel<false><<<grid, PQ2RowSplitMmaGemmSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, w.k, x.ne[1], out_row_stride, static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
    } else if (use_wmma1) {
        pq2_rowsplit_wmma_kernel<<<grid, PQ2RowSplitWmmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, w.k, x.ne[1], out_row_stride, static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
    } else {
        pq2_rowsplit_wmma2_kernel<<<grid, PQ2RowSplitWmma2Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            w.n, w.k, x.ne[1], out_row_stride, static_cast<std::int32_t>(w.qtype == QType::T2_G128_FP16));
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_ternary_gemm_t8(const Tensor& x, const Weight& w, Tensor& out,
                            std::int32_t out_row_stride, cudaStream_t stream) {
    // The speculative verify pass runs T = draft + 1 (2..4 here). Prefill arrives with T >= 128
    // and is carried by the same kernel, at ceil(T/8) weight passes. The reference tiled kernel
    // wastes five of its eight token slots at verify size and needs a 128-thread CTA plus seven
    // barriers per output row, which cost more than the whole decode step it was verifying.
    // MEASURED (A/B, T=512): MMA path 16.1 t/s vs reference path 22.1 t/s 鈥?the 8-token
    // tile's per-tile LUT decode makes it SLOWER than the reference for prefill, so the
    // MMA kernel is restricted to the verify/decode regime (T < 16) and prefill uses the
    // reference tiled path. Revisit after the reference kernel gains token batching.
    // MEASURED (probe/bench_small_t.cu, gfx1201, real model shapes, T=2):
    //   small_t   5.826 ms ( 8 GB/s of weight traffic)   <- 16-token block, 4 warps/CTA,
    //   wmma2     0.217 ms (218 GB/s)                      57 KB static smem => 1 CTA/SM
    //   gemv_tile 0.227 ms (209 GB/s)                    = 4 warps resident per SM. It is
    //                                                      not the token padding that kills
    //                                                      it, it is the occupancy.
    // T=3: 6.029 / 0.227. T=4: 5.711 / 0.214. So the whole 2..15 window now rides the
    // same register-decoded WMMA kernel prefill uses; the 16-token-block kernel stays
    // reachable with NINFER_TERNARY_SMALLT=1 as the A/B reference.
    // The verify-pass MLP alone went 9.34 ms/layer -> 0.38 ms/layer.
    static const bool force_small_t = [] {
        const char* v = std::getenv("NINFER_TERNARY_SMALLT");
        return v != nullptr && v[0] == '1';
    }();
    if (ternary_mma_enabled() && mma_admits(w, out_row_stride) && x.ne[1] >= 2 && x.ne[1] < 16 &&
        force_small_t) {
        TernaryTraceScope trace("mma_small_t", w.n, w.k, x.ne[1]);
        launch_pq2_mma_small_t(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    // PREFILL (T >= 16): the original-NInfer-style 2D-tiled MMA kernel (ported from
    // q4_rowsplit_gemm_mma.cuh). Validated against a CPU oracle by probe/prefill_probe.cu
    // (12/12 cases at bf16 half-ulp, incl. partial blocks / padded stride / K=5120).
    // Small-T verify (T = 2..15) joins it: the same kernel, grid_y = 1.
    if (ternary_mma_enabled() && mma_admits(w, out_row_stride) && x.ne[1] >= 2) {
        TernaryTraceScope trace("mma_prefill", w.n, w.k, x.ne[1]);
        launch_pq2_mma_prefill(x, w, out, out_row_stride, stream);
        return;
    }
    if (gemv_admits(x, w, 4)) {
        TernaryTraceScope trace("gemv_tile", w.n, w.k, x.ne[1]);
        launch_pq2_gemv_tile(x, w, out, out_row_stride, x.ne[1], stream);
        return;
    }
    // PREFILL BATCHING (original-NInfer-style 2D tiling, ported from
    // q4_rowsplit_gemm_mma.cuh): decode-to-smem + ldmatrix fragments + mma.
    // kTileT sweep note: the reference kernel at kTileT 16/64 collapses to ~4 t/s
    // (accumulator spills), so prefill goes through the dedicated 2D-tiled kernel.
    {
        TernaryTraceScope trace("by_qtype8", w.n, w.k, x.ne[1]);
        launch_by_qtype<8>(x, w, out, out_row_stride, stream);
    }
}


} // namespace ninfer::ops::detail

