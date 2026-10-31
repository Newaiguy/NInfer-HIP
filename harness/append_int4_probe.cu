
// Decisive probe: does the engine's own int4 append kernel write exactly what the codec spec says?
// Then feed that cache to the int4 prompt-WMMA attention kernel and compare against a CPU oracle.
#include "ops/common/math.cuh"
#include "ops/kv_cache/append/kernel.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16_wmma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ninfer::ops;
using Geometry = CausalD256H24Kv4;

static constexpr int D = 256;
static constexpr int Page = 64;
static constexpr int KVH = Geometry::KVHeads;
static constexpr int QH = Geometry::QHeads;

static std::uint16_t host_bf16(float v) {
    std::uint32_t bits; std::memcpy(&bits, &v, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<std::uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}
static float host_bf16_to_f(std::uint16_t h) {
    std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
static std::uint16_t host_f16(float v) { __half h = __float2half_rn(v); std::uint16_t o; std::memcpy(&o, &h, 2); return o; }
static float host_f16_to_f(std::uint16_t h) { __half hh; std::memcpy(&hh, &h, 2); return __half2float(hh); }
static int nib_signed(unsigned nib) { return static_cast<int>(nib ^ 8u) - 8; }

static unsigned rng_state;
static unsigned rng() { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }
static float frand() { return (float(int(rng() % 2001) - 1000) / 500.0f); }

static void wht256(float* v) {
    for (int s = 1; s < 256; s <<= 1) {
        for (int i = 0; i < 256; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                const float a = v[i + j], b = v[i + j + s];
                v[i + j] = a + b; v[i + j + s] = a - b;
            }
        }
    }
    for (int i = 0; i < 256; ++i) { v[i] *= (1.0f / 16.0f); }
}

static void host_nearest_d8(const float in[8], float out[8]) {
    float r[8];
    int parity = 0;
    float worst = -1.0f;
    int wi = 0;
    for (int i = 0; i < 8; ++i) {
        r[i] = std::nearbyint(in[i]);
        parity ^= (static_cast<int>(r[i]) & 1);
        const float e = std::fabs(in[i] - r[i]);
        if (e > worst || (e == worst && i < wi)) { worst = e; wi = i; }
    }
    if (parity != 0) { r[wi] += (in[wi] >= r[wi]) ? 1.0f : -1.0f; }
    for (int i = 0; i < 8; ++i) { out[i] = r[i]; }
}
static float host_sq_dist(const float x[8], const float p[8]) {
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) { const float d = x[i] - p[i]; sum += d * d; }
    return sum;
}
static void host_e8_octet(const float in[8], float out[8]) {
    float ip[8];
    host_nearest_d8(in, ip);
    float shifted[8], cp[8];
    for (int i = 0; i < 8; ++i) { shifted[i] = in[i] - 0.5f; }
    host_nearest_d8(shifted, cp);
    for (int i = 0; i < 8; ++i) { cp[i] += 0.5f; }
    const bool coset = host_sq_dist(in, cp) < host_sq_dist(in, ip);
    for (int i = 0; i < 8; ++i) { out[i] = coset ? cp[i] : ip[i]; }
}

struct Stats { int bad_k = 0, bad_v = 0, bad_ks = 0, bad_vs = 0, max_dk = 0, max_dv = 0; };

static bool run(int T) {
    rng_state = 20261027u + unsigned(T);
    const int pages = (T + Page - 1) / Page;
    std::vector<std::int32_t> table(pages);
    for (int p = 0; p < pages; ++p) { table[p] = pages - 1 - p; }  // non-identity permutation
    const auto phys = [&](int p) { return table[p]; };  // follow the same table the kernel reads

    std::vector<std::uint16_t> k_in(std::size_t(T) * KVH * D), v_in(std::size_t(T) * KVH * D);
    bool real_data = false;
    {
        std::FILE* fk = std::fopen("D:/ninfer-hip-migrate/build/kvdump/k_0.bin", "rb");
        std::FILE* fv = std::fopen("D:/ninfer-hip-migrate/build/kvdump/v_0.bin", "rb");
        std::FILE* ft = std::fopen("D:/ninfer-hip-migrate/build/kvdump/tables_0.bin", "rb");
        if (fk != nullptr && fv != nullptr && ft != nullptr && T == 522) {
            std::fread(k_in.data(), 2, k_in.size(), fk);
            std::fread(v_in.data(), 2, v_in.size(), fv);
            std::fread(table.data(), 4, table.size(), ft);
            real_data = true;
            std::printf("  T=%d using REAL engine k/v dumps, table[0..8]=", T);
            for (int i = 0; i < 9; ++i) { std::printf("%d ", table[i]); }
            std::printf("\n");
        }
        if (fk) { std::fclose(fk); }
        if (fv) { std::fclose(fv); }
        if (ft) { std::fclose(ft); }
    }
    if (!real_data) {
        for (auto& x : k_in) { x = host_bf16(frand()); }
        for (auto& x : v_in) { x = host_bf16(frand()); }
    }

    const std::size_t row_bytes = 128;
    std::vector<std::uint8_t> kc(std::size_t(pages) * KVH * Page * row_bytes, 0);
    std::vector<std::uint8_t> vc(std::size_t(pages) * KVH * Page * row_bytes, 0);
    std::vector<std::uint16_t> ks(std::size_t(pages) * KVH * Page * 4, 0);
    std::vector<std::uint16_t> vs(std::size_t(pages) * KVH * Page * 8, 0);
    std::vector<float> k_deq(std::size_t(pages) * KVH * Page * D, 0.0f);
    std::vector<float> v_deq(std::size_t(pages) * KVH * Page * D, 0.0f);

    for (int page = 0; page < pages; ++page) {
        for (int head = 0; head < KVH; ++head) {
            for (int po = 0; po < Page; ++po) {
                const int tok = page * Page + po;
                if (tok >= T) { continue; }
                const std::size_t src = std::size_t(D) * (head + KVH * tok);
                float krow[D];
                for (int d = 0; d < D; ++d) { krow[d] = host_bf16_to_f(k_in[src + d]); }
                wht256(krow);
                const std::size_t kb = (std::size_t(phys(page)) * KVH + head) * Page * row_bytes +
                                       std::size_t(po) * row_bytes;
                const std::size_t vb = kb;
                for (int g = 0; g < 4; ++g) {
                    float amax = 0.0f;
                    for (int d = 0; d < 64; ++d) { amax = std::max(amax, std::fabs(krow[g * 64 + d])); }
                    const std::uint16_t sc = host_f16(amax > 0.0f ? amax / 7.0f : 0.0f);
                    const float scf = host_f16_to_f(sc);
                    const float inv = scf > 0.0f ? 1.0f / scf : 0.0f;
                    ks[(std::size_t(phys(page)) * KVH + head) * Page * 4 + std::size_t(po) * 4 + g] = sc;
                    for (int j = 0; j < 8; ++j) {
                        float in8[8], out8[8];
                        for (int i = 0; i < 8; ++i) { out8[i] = inv == 0.0f ? 0.0f : krow[g * 64 + j * 8 + i] * inv; }
                        for (int i = 0; i < 8; ++i) {
                            int q = static_cast<int>(std::nearbyint(out8[i]));
                            q = std::max(-8, std::min(7, q));
                            const unsigned nib = static_cast<unsigned>(q) & 0x0fu;
                            const std::size_t byte = kb + std::size_t(g) * 32 + std::size_t(j) * 4 + (i >> 1);
                            if ((i & 1) == 0) { kc[byte] = static_cast<std::uint8_t>((kc[byte] & 0xf0u) | nib); }
                            else              { kc[byte] = static_cast<std::uint8_t>((kc[byte] & 0x0fu) | (nib << 4)); }
                            k_deq[(std::size_t(page) * KVH + head) * Page * D + std::size_t(po) * D +
                                  g * 64 + j * 8 + i] = float(q) * scf;
                        }
                    }
                }
                for (int g = 0; g < 8; ++g) {
                    float amax = 0.0f;
                    for (int d = 0; d < 32; ++d) { amax = std::max(amax, std::fabs(host_bf16_to_f(v_in[src + g * 32 + d]))); }
                    const std::uint16_t sc = host_f16(amax > 0.0f ? amax / 7.0f : 0.0f);
                    const float scf = host_f16_to_f(sc);
                    const float inv = scf > 0.0f ? 1.0f / scf : 0.0f;
                    vs[(std::size_t(phys(page)) * KVH + head) * Page * 8 + std::size_t(po) * 8 + g] = sc;
                    for (int d = 0; d < 32; ++d) {
                        const float x = host_bf16_to_f(v_in[src + g * 32 + d]);
                        int q = inv == 0.0f ? 0 : static_cast<int>(std::nearbyint(x * inv));
                        q = std::max(-7, std::min(7, q));
                        const unsigned nib = static_cast<unsigned>(q) & 0x0fu;
                        const std::size_t byte = vb + std::size_t(g) * 16 + (d >> 1);
                        if ((d & 1) == 0) { vc[byte] = static_cast<std::uint8_t>((vc[byte] & 0xf0u) | nib); }
                        else              { vc[byte] = static_cast<std::uint8_t>((vc[byte] & 0x0fu) | (nib << 4)); }
                        v_deq[(std::size_t(page) * KVH + head) * Page * D + std::size_t(po) * D +
                              g * 32 + d] = float(q) * scf;
                    }
                }
            }
        }
    }

    std::uint16_t *d_k, *d_v;
    std::int32_t *d_pos, *d_rows, *d_table;
    std::uint8_t *d_kc, *d_vc;
    std::uint16_t *d_ks, *d_vs;
    cudaMalloc(&d_k, k_in.size() * 2);
    cudaMalloc(&d_v, v_in.size() * 2);
    cudaMalloc(&d_pos, 4);
    cudaMalloc(&d_rows, 4);
    cudaMalloc(&d_table, table.size() * 4);
    cudaMalloc(&d_kc, kc.size());
    cudaMalloc(&d_vc, vc.size());
    cudaMalloc(&d_ks, ks.size() * 2);
    cudaMalloc(&d_vs, vs.size() * 2);
    cudaMemcpy(d_k, k_in.data(), k_in.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_in.data(), v_in.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_kc, 0, kc.size());
    cudaMemset(d_vc, 0, vc.size());
    cudaMemset(d_ks, 0, ks.size() * 2);
    cudaMemset(d_vs, 0, vs.size() * 2);
    const std::int32_t zero = 0;
    cudaMemcpy(d_pos, &zero, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, &zero, 4, cudaMemcpyHostToDevice);

    const PagedKVBatchMetadata<false> meta{d_table, nullptr, d_rows, pages};
    const int fill_grid = (T * KVH + 7) / 8;
    kv_cache_append_full_i8v_kernel<Geometry, PagedKVBatchMetadata<false>, true, KvKeyCoding::Int4E8>
        <<<fill_grid, 256>>>(
            reinterpret_cast<const __nv_bfloat16*>(d_k), reinterpret_cast<const __nv_bfloat16*>(d_v),
            d_pos, meta, reinterpret_cast<std::int8_t*>(d_kc), reinterpret_cast<std::int8_t*>(d_vc),
            reinterpret_cast<__half*>(d_ks), reinterpret_cast<__half*>(d_vs), T);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("  T=%d append launch error: %s\n", T, cudaGetErrorString(err)); return false; }

    std::vector<std::uint8_t> kc_got(kc.size()), vc_got(vc.size());
    std::vector<std::uint16_t> ks_got(ks.size()), vs_got(vs.size());
    cudaMemcpy(kc_got.data(), d_kc, kc.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(vc_got.data(), d_vc, vc.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ks_got.data(), d_ks, ks.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(vs_got.data(), d_vs, vs.size() * 2, cudaMemcpyDeviceToHost);

    Stats st;
    for (std::size_t i = 0; i < kc.size(); ++i) {
        if (kc_got[i] == kc[i]) { continue; }
        const int d0 = nib_signed(kc_got[i] & 0x0fu) - nib_signed(kc[i] & 0x0fu);
        const int d1 = nib_signed(kc_got[i] >> 4) - nib_signed(kc[i] >> 4);
        if (d0 != 0) { ++st.bad_k; st.max_dk = std::max(st.max_dk, std::abs(d0)); }
        if (d1 != 0) { ++st.bad_k; st.max_dk = std::max(st.max_dk, std::abs(d1)); }
    }
    for (std::size_t i = 0; i < vc.size(); ++i) {
        if (vc_got[i] == vc[i]) { continue; }
        const int d0 = nib_signed(vc_got[i] & 0x0fu) - nib_signed(vc[i] & 0x0fu);
        const int d1 = nib_signed(vc_got[i] >> 4) - nib_signed(vc[i] >> 4);
        if (d0 != 0) { ++st.bad_v; st.max_dv = std::max(st.max_dv, std::abs(d0)); }
        if (d1 != 0) { ++st.bad_v; st.max_dv = std::max(st.max_dv, std::abs(d1)); }
    }
    for (std::size_t i = 0; i < ks.size(); ++i) { if (ks_got[i] != ks[i]) { ++st.bad_ks; } }
    for (std::size_t i = 0; i < vs.size(); ++i) { if (vs_got[i] != vs[i]) { ++st.bad_vs; } }
    std::printf("  T=%4d appendbytes: K code mism=%d (max |dcode|=%d of %d), V code mism=%d (max=%d), "
                "K scale mism=%d/%d, V scale mism=%d/%d\n",
                T, st.bad_k, st.max_dk, int(kc.size()), st.bad_v, st.max_dv,
                st.bad_ks, int(ks.size()), st.bad_vs, int(vs.size()));

    // ---- stage 2: run the real attention kernel on the cache just written ----
    std::vector<std::uint16_t> q_in(std::size_t(T) * QH * D);
    for (auto& x : q_in) { x = host_bf16(frand()); }
    __nv_bfloat16* d_q = nullptr;
    __nv_bfloat16* d_out = nullptr;
    std::vector<std::uint16_t> out(std::size_t(T) * QH * D);
    cudaMalloc(&d_q, q_in.size() * 2);
    cudaMalloc(&d_out, out.size() * 2);
    cudaMemcpy(d_q, q_in.data(), q_in.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0xEE, out.size() * 2);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const PagedKVDirectMetadata dmeta{d_table};
    const dim3 grid((unsigned)((T + kCausalPromptBr - 1) / kCausalPromptBr), (unsigned)QH);
    causal_attention_prompt_bf16_wmma_kernel<Geometry, PagedKVDirectMetadata, false, false, false, true>
        <<<grid, kCausalPromptThreads, kCausalPromptWmmaInt8SmemBytes>>>(
            d_q, reinterpret_cast<const __nv_bfloat16*>(d_kc), reinterpret_cast<const __half*>(d_vc),
            reinterpret_cast<const __half*>(d_ks), reinterpret_cast<const __half*>(d_vs),
            dmeta, d_pos, scale, d_out, T);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("  T=%d attn launch error: %s\n", T, cudaGetErrorString(err)); return false; }
    cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost);

    // CPU oracle on the dequantized planes (logical indexing)
    std::vector<float> qrot(std::size_t(T) * QH * D);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < QH; ++h) {
            float row[D];
            for (int d = 0; d < D; ++d) { row[d] = host_bf16_to_f(q_in[std::size_t(t) * QH * D + std::size_t(h) * D + d]); }
            wht256(row);
            float* dst = &qrot[std::size_t(t) * QH * D + std::size_t(h) * D];
            for (int g = 0; g < 4; ++g) {
                float amax = 0.0f;
                for (int d = 0; d < 64; ++d) { amax = std::max(amax, std::fabs(row[g * 64 + d])); }
                const std::uint16_t s16 = host_f16(amax > 0.0f ? amax / 127.0f : 0.0f);
                const float sf = host_f16_to_f(s16);
                const float inv = sf > 0.0f ? 1.0f / sf : 0.0f;
                for (int d = 0; d < 64; ++d) {
                    int q = inv == 0.0f ? 0 : static_cast<int>(std::nearbyint(row[g * 64 + d] * inv));
                    q = std::max(-127, std::min(127, q));
                    dst[g * 64 + d] = float(q) * sf * scale;
                }
            }
        }
    }
    double worst = 0.0; int bad = 0, wt = -1, wh = -1, wd = -1;
    std::vector<double> sc(T);
    for (int h = 0; h < QH; ++h) {
        const int kvh = h / Geometry::GroupSize;
        for (int t = 0; t < T; ++t) {
            double m = -1e30;
            for (int s = 0; s <= t; ++s) {
                const std::size_t base = (std::size_t(s / Page) * KVH + kvh) * Page * D +
                                         std::size_t(s % Page) * D;
                double acc = 0.0;
                for (int d = 0; d < D; ++d) {
                    acc += double(qrot[std::size_t(t) * QH * D + std::size_t(h) * D + d]) * k_deq[base + d];
                }
                sc[s] = acc; m = std::max(m, acc);
            }
            double l = 0.0;
            for (int s = 0; s <= t; ++s) { sc[s] = std::exp(sc[s] - m); l += sc[s]; }
            for (int d = 0; d < D; ++d) {
                double acc = 0.0;
                for (int s = 0; s <= t; ++s) {
                    const std::size_t base = (std::size_t(s / Page) * KVH + kvh) * Page * D +
                                             std::size_t(s % Page) * D;
                    acc += sc[s] * v_deq[base + d];
                }
                acc /= l;
                const double got = host_bf16_to_f(out[std::size_t(t) * QH * D + std::size_t(h) * D + d]);
                const double dd = std::fabs(got - acc);
                if (dd > worst) { worst = dd; wt = t; wh = h; wd = d; }
                if (dd > 0.02 + 0.05 * std::fabs(acc)) { ++bad; }
            }
        }
    }
    std::printf("  T=%4d attention: %d wrong, worst=%.4f (t=%d h=%d d=%d)\n", T, bad, worst, wt, wh, wd);
    cudaFree(d_q); cudaFree(d_out); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_pos); cudaFree(d_rows); cudaFree(d_table);
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ks); cudaFree(d_vs);
    return st.bad_k == 0 && st.bad_v == 0 && st.bad_ks == 0 && st.bad_vs == 0 && bad == 0;
}

int main() {
    bool ok = true;
    for (int T : {1, 63, 64, 65, 130, 522}) { ok = run(T) && ok; }
    std::printf(ok ? "APPEND+ATTN PROBE: ALL PASS\n" : "APPEND+ATTN PROBE: FAILURES PRESENT\n");
    return ok ? 0 : 1;
}
