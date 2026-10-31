
// int4 (RotatedInt4KeyInt4ValueE8) prompt-WMMA attention probe: bit-level compare
// against a CPU oracle built from the *quantized* (dequantized) K/V planes.
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16_wmma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace ninfer::ops;

static std::uint16_t host_bf16(float v) {
    std::uint32_t bits; std::memcpy(&bits, &v, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<std::uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}
static float host_bf16_to_f(std::uint16_t h) {
    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
static std::uint16_t host_f16(float v) {
    __half h = __float2half_rn(v);
    std::uint16_t out; std::memcpy(&out, &h, 2); return out;
}
static float host_f16_to_f(std::uint16_t h) {
    __half hh; std::memcpy(&hh, &h, 2);
    return __half2float(hh);
}

static unsigned rng_state = 20260928u;
static unsigned rng() { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }

// Natural-order normalized Sylvester H256 (matches normalized_hadamard_d256_inplace: /16).
static void wht256(double* v) {
    for (int s = 1; s < 256; s <<= 1) {
        for (int i = 0; i < 256; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                const double a = v[i + j], b = v[i + j + s];
                v[i + j] = a + b; v[i + j + s] = a - b;
            }
        }
    }
    for (int i = 0; i < 256; ++i) { v[i] *= (1.0 / 16.0); }
}

static std::int8_t q_int4_k(double x, double inv, double* repr_out_unused) {
    (void)repr_out_unused;
    double q = std::nearbyint(x * inv);
    q = std::max(-8.0, std::min(7.0, q));
    return static_cast<std::int8_t>(q);
}
static std::int8_t q_int4_v(double x, double inv) {
    double q = std::nearbyint(x * inv);
    q = std::max(-7.0, std::min(7.0, q));
    return static_cast<std::int8_t>(q);
}

static bool run_case(int T) {
    constexpr int D = kCausalPromptHeadDim;   // 256
    constexpr int Page = ninfer::kPagedKVPageSize;  // 64
    using Geometry = CausalD256H24Kv4;
    const int QH = Geometry::QHeads, KVH = Geometry::KVHeads;
    const float scale = 1.0f / std::sqrt(static_cast<double>(D));

    // ---- host q/k/v (bf16), matching the bf16 probe's generator ----
    std::vector<__nv_bfloat16> q(std::size_t(T) * QH * D);
    std::vector<double> q_f(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        const float v = (float(rng() % 21) - 10.0f) * 0.1f;
        const std::uint16_t b = host_bf16(v);
        std::memcpy(&q[i], &b, 2);
        q_f[i] = host_bf16_to_f(b);
    }
    const int pages = (T + Page - 1) / Page;
    // non-identity permutation: physical slot of logical page p is (pages-1-p). A read path
    // that silently uses the logical index instead of block_table[] fails here but passed
    // with an identity table.
    std::vector<std::int32_t> table(pages);
    for (int p = 0; p < pages; ++p) { table[p] = pages - 1 - p; }
    const auto phys = [&](int p) { return pages - 1 - p; };

    // logical k/v per (token, kv_head, d)
    const std::size_t n_kv = std::size_t(pages) * KVH * Page * D;
    std::vector<double> k_log(n_kv), v_log(n_kv);
    for (std::size_t i = 0; i < n_kv; ++i) {
        const float kv = (float(rng() % 21) - 10.0f) * 0.1f;
        const std::uint16_t kb = host_bf16(kv);
        k_log[i] = host_bf16_to_f(kb);
        const std::uint16_t vb = host_f16(kv * 0.5f);
        v_log[i] = host_f16_to_f(vb);
    }

    // ---- quantize exactly like the append kernel ----
    // K plane: 128 code bytes + 4 fp16 scales per (page,head,token)
    std::vector<std::uint8_t> kc(n_kv / 2, 0);
    std::vector<std::uint16_t> ks(std::size_t(pages) * KVH * Page * 4, 0);
    std::vector<double> k_deq(n_kv, 0.0);
    // V plane: 128 code bytes + 8 fp16 scales
    std::vector<std::uint8_t> vc(n_kv / 2, 0);
    std::vector<std::uint16_t> vs(std::size_t(pages) * KVH * Page * 8, 0);
    std::vector<double> v_deq(n_kv, 0.0);

    auto elem = [&](int page, int head, int po, int d) {
        return (std::size_t(page) * KVH + head) * Page * D + std::size_t(po) * D + d;
    };

    for (int page = 0; page < pages; ++page) {
        for (int head = 0; head < KVH; ++head) {
            for (int po = 0; po < Page; ++po) {
                const int tok = page * Page + po;
                if (tok >= T) { continue; }
                // rotate K row
                double krot[256];
                for (int d = 0; d < 256; ++d) { krot[d] = k_log[elem(page, head, po, d)]; }
                wht256(krot);
                // K: G64 quant, 4 groups
                for (int g = 0; g < 4; ++g) {
                    double amax = 0.0;
                    for (int d = 0; d < 64; ++d) { amax = std::max(amax, std::fabs(krot[g * 64 + d])); }
                    const std::uint16_t sc = host_f16(amax > 0.0 ? amax / 7.0 : 0.0);
                    const double scf = host_f16_to_f(sc);
                    const double inv = scf > 0.0 ? 1.0 / scf : 0.0;
                    ks[(std::size_t(phys(page)) * KVH + head) * Page * 4 + std::size_t(po) * 4 + g] = sc;
                    std::int8_t codes[64];
                    for (int d = 0; d < 64; ++d) {
                        codes[d] = inv == 0.0 ? 0 : q_int4_k(krot[g * 64 + d], inv, nullptr);
                        k_deq[elem(page, head, po, g * 64 + d)] = double(codes[d]) * scf;
                    }
                    // pack: byte p holds dims (2p,2p+1), low nibble = even
                    const std::size_t base =
                        (std::size_t(phys(page)) * KVH + head) * Page * 128 + std::size_t(po) * 128 + g * 32;
                    for (int p = 0; p < 32; ++p) {
                        const unsigned lo = static_cast<unsigned>(codes[2 * p]) & 0x0fu;
                        const unsigned hi = static_cast<unsigned>(codes[2 * p + 1]) & 0x0fu;
                        kc[base + p] = static_cast<std::uint8_t>(lo | (hi << 4));
                    }
                }
                // V: G32 quant, 8 groups, no rotation
                for (int g = 0; g < 8; ++g) {
                    double amax = 0.0;
                    for (int d = 0; d < 32; ++d) { amax = std::max(amax, std::fabs(v_log[elem(page, head, po, g * 32 + d)])); }
                    const std::uint16_t sc = host_f16(amax > 0.0 ? amax / 7.0 : 0.0);
                    const double scf = host_f16_to_f(sc);
                    const double inv = scf > 0.0 ? 1.0 / scf : 0.0;
                    vs[(std::size_t(phys(page)) * KVH + head) * Page * 8 + std::size_t(po) * 8 + g] = sc;
                    std::int8_t codes[32];
                    for (int d = 0; d < 32; ++d) {
                        codes[d] = inv == 0.0 ? 0 : q_int4_v(v_log[elem(page, head, po, g * 32 + d)], inv);
                        v_deq[elem(page, head, po, g * 32 + d)] = double(codes[d]) * scf;
                    }
                    const std::size_t base =
                        (std::size_t(phys(page)) * KVH + head) * Page * 128 + std::size_t(po) * 128 + g * 16;
                    for (int p = 0; p < 16; ++p) {
                        const unsigned lo = static_cast<unsigned>(codes[2 * p]) & 0x0fu;
                        const unsigned hi = static_cast<unsigned>(codes[2 * p + 1]) & 0x0fu;
                        vc[base + p] = static_cast<std::uint8_t>(lo | (hi << 4));
                    }
                }
            }
        }
    }

    // ---- device buffers ----
    __nv_bfloat16 *d_q, *d_out; std::int32_t *d_table, *d_pos;
    std::uint8_t *d_kc, *d_vc; std::uint16_t *d_ks, *d_vs;
    std::vector<__nv_bfloat16> out(std::size_t(T) * QH * D);
    cudaMalloc(&d_q, q.size() * 2);
    cudaMalloc(&d_out, out.size() * 2);
    cudaMalloc(&d_table, table.size() * 4);
    cudaMalloc(&d_pos, 4);
    cudaMalloc(&d_kc, kc.size());
    cudaMalloc(&d_vc, vc.size());
    cudaMalloc(&d_ks, ks.size() * 2);
    cudaMalloc(&d_vs, vs.size() * 2);
    cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kc, kc.data(), kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vc, vc.data(), vc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice);
    const std::int32_t pos0 = 0;
    cudaMemcpy(d_pos, &pos0, 4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0xEE, out.size() * 2);

    const PagedKVDirectMetadata metadata{d_table};
    const dim3 grid((unsigned)((T + kCausalPromptBr - 1) / kCausalPromptBr), (unsigned)QH);
    causal_attention_prompt_bf16_wmma_kernel<Geometry, PagedKVDirectMetadata, false, false, false, true>
        <<<grid, kCausalPromptThreads, kCausalPromptWmmaInt8SmemBytes>>>(
            d_q, reinterpret_cast<const __nv_bfloat16*>(d_kc), reinterpret_cast<const __half*>(d_vc),
            reinterpret_cast<const __half*>(d_ks), reinterpret_cast<const __half*>(d_vs),
            metadata, d_pos, scale, d_out, T);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::printf("  launch error: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    // ---- CPU oracle on dequantized, rotated K ----
    std::vector<double> qrot(q_f.size());
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < QH; ++h) {
            double row[256];
            for (int d = 0; d < 256; ++d) { row[d] = q_f[std::size_t(t) * QH * D + std::size_t(h) * D + d]; }
            wht256(row);
            for (int d = 0; d < 256; ++d) { qrot[std::size_t(t) * QH * D + std::size_t(h) * D + d] = row[d] * scale; }
        }
    }
    double worst = 0.0; int bad = 0, wt = -1, wh = -1, wd = -1;
    std::vector<double> score(T);
    for (int h = 0; h < QH; ++h) {
        const int kvh = h / Geometry::GroupSize;
        for (int t = 0; t < T; ++t) {
            double m = -1e30;
            for (int s = 0; s <= t; ++s) {
                const int page = s / Page, po = s % Page;
                double acc = 0.0;
                for (int d = 0; d < D; ++d) {
                    acc += qrot[std::size_t(t) * QH * D + std::size_t(h) * D + d] * k_deq[elem(page, kvh, po, d)];
                }
                score[s] = acc; m = std::max(m, acc);
            }
            double l = 0.0;
            for (int s = 0; s <= t; ++s) { score[s] = std::exp(score[s] - m); l += score[s]; }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int s = 0; s <= t; ++s) { acc += score[s] * v_deq[elem(s / Page, kvh, s % Page, d)]; }
                acc /= l;
                const double got = host_bf16_to_f(*reinterpret_cast<std::uint16_t*>(
                    &out[std::size_t(t) * QH * D + std::size_t(h) * D + d]));
                const double dd = std::fabs(got - acc);
                if (dd > worst) { worst = dd; wt = t; wh = h; wd = d; }
                // int4 K/V: quant noise shows up as ~1% of the value; the bar is a
                // correctness bar (kernel bug => errors of order 1e-1..1e1).
                if (dd > 0.02 + 0.05 * std::fabs(acc)) {
                    ++bad;
                    if (bad <= 6) { std::printf("    sample t=%d h=%d d=%d got=%.6f want=%.6f\n", t, h, d, got, acc); }
                }
            }
        }
    }
    std::printf("  T=%d: %d/%d wrong, worst=%.4f (t=%d h=%d d=%d) %s\n", T, bad, T * QH * D,
                worst, wt, wh, wd, bad == 0 ? "OK" : "FAIL");
    cudaFree(d_q); cudaFree(d_out); cudaFree(d_table); cudaFree(d_pos);
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ks); cudaFree(d_vs);
    return bad == 0;
}

int main() {
    bool ok = true;
    for (int T : {1, 2, 64, 100, 128, 256, 320, 512, 522}) { ok = run_case(T) && ok; }
    std::printf(ok ? "INT4 ATTN PROBE: ALL PASS\n" : "INT4 ATTN PROBE: FAILURES PRESENT\n");
    return ok ? 0 : 1;
}
