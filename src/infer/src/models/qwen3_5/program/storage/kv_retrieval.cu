#include "models/qwen3_5/program/storage/kv_retrieval.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::detail {
namespace {

// This tree has no project-wide CUDA error-check macro, and the retrieval buffers are the only
// thing this file allocates, so a local helper keeps the failure path explicit.
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("kv-retrieve: ") + what + " failed: " +
                                 cudaGetErrorString(status));
    }
}

#define KV_CHECK(call) check((call), #call)

}  // namespace
}  // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {
namespace {

constexpr int kPageElems   = kRetrievalKVHeads * kRetrievalHeadDim;  // 1024 values per layer
constexpr int kLayerStride = kRetrievalSubChunks * kPageElems;       // per layer
constexpr int kPageHalf    = kRetrievalLayers * kPageElems;          // min block, max follows
constexpr int kPageStride  = 3 * kPageHalf;                          // mean, min, max per channel
constexpr int kScoreUnits  = kRetrievalLayers * kRetrievalKVHeads;   // one score per (layer, head)
constexpr int kQueryGroup  = kRetrievalQueryHeads / kRetrievalKVHeads;

// Folds one chunk into the per-page pre-RoPE key sums.
//
// Grid is one block per page touched by this chunk, so a page is owned by exactly one block for
// a given layer and no atomics are needed; the 1024 summary values are indexed directly by the
// thread, so there is no reduction either. Tokens of a page are contiguous inside a chunk, and a
// chunk spans at most a handful of pages, so the scan below is a few hundred comparisons.
__global__ void kmeans_page_kernel(const __nv_bfloat16* __restrict__ keys, int layer,
                                   const std::int32_t* __restrict__ positions, int tokens,
                                   std::uint32_t first_page, float* __restrict__ ring,
                                   std::uint32_t* __restrict__ tags,
                                   std::uint32_t* __restrict__ counts) {
    const std::uint32_t page = first_page + static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t slot = page % static_cast<std::uint32_t>(kRetrievalRingPages);
    const int tid            = threadIdx.x;

    __shared__ int per_sub[kRetrievalSubChunks];
    __shared__ int fresh;
    if (tid == 0) {
        for (int sub = 0; sub < kRetrievalSubChunks; ++sub) { per_sub[sub] = 0; }
        for (int t = 0; t < tokens; ++t) {
            const std::uint32_t pos = static_cast<std::uint32_t>(positions[t]);
            if (pos / kRetrievalPageTokens != page) { continue; }
            ++per_sub[(pos % kRetrievalPageTokens) / kRetrievalSubTokens];
        }
        fresh = (tags[slot] != page + 1U) ? 1 : 0;
        if (fresh != 0) {
            tags[slot] = page + 1U;
            for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
                counts[slot * kRetrievalSubChunks + sub] = 0;
            }
        }
    }
    __syncthreads();
    if (tid == 0 && page == 118U) {
        printf("[kv-cap] page=%u tokens=%d sub0=%d fresh=%d\n", page, tokens, per_sub[0], fresh);
    }
    if (tid < kPageElems) {
        float lo = 1.0e30F;
        float hi = -1.0e30F;
        float sum = 0.0F;
        for (int t = 0; t < tokens; ++t) {
            const std::uint32_t pos = static_cast<std::uint32_t>(positions[t]);
            if (pos / kRetrievalPageTokens != page) { continue; }
            const float value =
                __bfloat162float(keys[static_cast<std::size_t>(t) * kPageElems + tid]);
            sum += value;
            lo = value < lo ? value : lo;
            hi = value > hi ? value : hi;
        }
        if (lo > hi) { return; }
        const std::size_t base = static_cast<std::size_t>(slot) * kPageStride +
                                 static_cast<std::size_t>(layer) * kPageElems + tid;
        // A page can straddle two chunks, so both ends of the extent and the running sum have to
        // fold into whatever an earlier chunk already stored.
        if (fresh != 0) {
            ring[base]                  = sum;
            ring[base + kPageHalf]      = lo;
            ring[base + 2 * kPageHalf]  = hi;
        } else {
            ring[base]                 += sum;
            ring[base + kPageHalf]      = ring[base + kPageHalf] < lo ? ring[base + kPageHalf] : lo;
            ring[base + 2 * kPageHalf]  = ring[base + 2 * kPageHalf] > hi ? ring[base + 2 * kPageHalf] : hi;
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
            counts[slot * kRetrievalSubChunks + sub] += static_cast<std::uint32_t>(per_sub[sub]);
        }
    }
}

// Folds this chunk's pre-RoPE query rows into the query summary. Query heads are 6:1 grouped
// onto the kv heads, and the score below is a dot product, so summing the six query heads of a
// group before the dot is equivalent to summing six dot products afterwards.
__global__ void kquery_page_kernel(const __nv_bfloat16* __restrict__ query, int layer,
                                   int tokens, float* __restrict__ summary) {
    const int tid = threadIdx.x;  // [0, kPageElems)
    if (tid >= kPageElems) { return; }
    const int kv_head  = tid / kRetrievalHeadDim;
    const int dim      = tid % kRetrievalHeadDim;
    float sum          = 0.0F;
    for (int t = 0; t < tokens; ++t) {
        for (int g = 0; g < kQueryGroup; ++g) {
            const int q_head = kv_head * kQueryGroup + g;
            const std::size_t index =
                static_cast<std::size_t>(q_head) * kRetrievalHeadDim + dim +
                static_cast<std::size_t>(t) * (kRetrievalQueryHeads * kRetrievalHeadDim);
            sum += __bfloat162float(query[index]);
        }
    }
    summary[static_cast<std::size_t>(layer) * kPageElems + tid] += sum;
}

// One block per candidate page: sums the per-layer dot products into a single score.
__device__ int g_kv_use_bound = 0;

__global__ void kscore_page_kernel(const __nv_bfloat16* __restrict__ page_means,
                                   const float* __restrict__ query, const float* __restrict__ mean,
                                   float* __restrict__ scores, int entries) {
    extern __shared__ float shared[];
    const int entry = blockIdx.x;
    if (entry >= entries) { return; }
    const __nv_bfloat16* page = page_means + static_cast<std::size_t>(entry) * kPageStride;
    const int tid             = threadIdx.x;
    for (int layer = 0; layer < kRetrievalLayers; ++layer) {
        for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
            float local = 0.0F;
            if (tid < kPageElems) {
                const std::size_t offset = static_cast<std::size_t>(sub) * kLayerStride +
                                           static_cast<std::size_t>(layer) * kPageElems + tid;
                const float q    = query[static_cast<std::size_t>(layer) * kPageElems + tid];
                const float mean = __bfloat162float(page[offset]);
                const float lo   = __bfloat162float(page[offset + kPageHalf]);
                const float hi   = __bfloat162float(page[offset + 2 * kPageHalf]);
                // Two views of the same page: the averaged key (good when the whole page is
                // relevant) and the per-channel extent bound (good when only a few channels fire,
                // as happens in a long document). Keep whichever is stronger.
                const float averaged = q * mean;
                const float bound    = q * (q >= 0.0F ? hi : lo);
                local = g_kv_use_bound != 0 ? fmaxf(averaged, bound) : averaged;
            }
            shared[tid] = local;
            __syncthreads();
            for (int stride = kRetrievalHeadDim / 2; stride > 0; stride >>= 1) {
                const int lane = tid % kRetrievalHeadDim;
                if (lane < stride) { shared[tid] += shared[tid + stride]; }
                __syncthreads();
            }
            if ((tid % kRetrievalHeadDim) == 0) {
                const int head = tid / kRetrievalHeadDim;
                float& slot    = scores[static_cast<std::size_t>(entry) * kScoreUnits +
                                     layer * kRetrievalKVHeads + head];
                if (sub == 0 || shared[tid] > slot) { slot = shared[tid]; }
            }
            __syncthreads();
        }
    }
}

__global__ void knormalize_query_kernel(float* __restrict__ query) {
    extern __shared__ float shared[];
    const int layer = blockIdx.x;
    const int tid   = threadIdx.x;
    float* layer_query = query + static_cast<std::size_t>(layer) * kPageElems;
    const float value  = layer_query[tid];
    shared[tid]        = value * value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { shared[tid] += shared[tid + stride]; }
        __syncthreads();
    }
    const float scale = 1.0F / (std::sqrt(shared[0]) + 1.0e-6F);
    layer_query[tid]  = value * scale;
}

constexpr int kLastQueryTokens = 1;

// Candidate-set mean: the dominant direction every page shares. Subtracting it before the dot
// keeps "typical of the document" from outranking "relevant to the question".
__global__ void kmean_page_kernel(const __nv_bfloat16* __restrict__ page_means, int entries,
                                  float* __restrict__ mean) {
    for (int idx = threadIdx.x; idx < kPageStride; idx += blockDim.x) {
        float sum = 0.0F;
        for (int entry = 0; entry < entries; ++entry) {
            sum += __bfloat162float(page_means[static_cast<std::size_t>(entry) * kPageStride + idx]);
        }
        mean[idx] = sum / static_cast<float>(entries);
    }
}

__global__ void klast_query_kernel(const __nv_bfloat16* __restrict__ query, int tokens, int layer,
                                   float* __restrict__ out) {
    const int tid = threadIdx.x;
    if (tid >= kPageElems || tokens <= 0) { return; }
    const int kv_head = tid / kRetrievalHeadDim;
    const int dim     = tid % kRetrievalHeadDim;
    const int t_begin = tokens > kLastQueryTokens ? tokens - kLastQueryTokens : 0;
    float sum         = 0.0F;
    for (int t = t_begin; t < tokens; ++t) {
        for (int g = 0; g < kQueryGroup; ++g) {
            const int q_head        = kv_head * kQueryGroup + g;
            const std::size_t index = static_cast<std::size_t>(q_head) * kRetrievalHeadDim + dim +
                                      static_cast<std::size_t>(t) * (kRetrievalQueryHeads * kRetrievalHeadDim);
            sum += __bfloat162float(query[index]);
        }
    }
    out[static_cast<std::size_t>(layer) * kPageElems + tid] =
        sum / static_cast<float>(tokens - t_begin);
}

bool retrieval_enabled() {
    static const bool value = [] {
        const char* env = std::getenv("NINFER_KV_RETRIEVE");
        return env != nullptr && env[0] == '1';
    }();
    return value;
}

}  // namespace

bool KVRetrieval::enabled() { return retrieval_enabled(); }

KVRetrieval* KVRetrieval::instance() noexcept {
    if (!retrieval_enabled()) { return nullptr; }
    static KVRetrieval singleton;
    return &singleton;
}

KVRetrieval::KVRetrieval() = default;

void KVRetrieval::ensure_allocated() {
    if (means_ != nullptr) { return; }
    const std::size_t ring_bytes =
        static_cast<std::size_t>(kRetrievalRingPages) * kPageStride * sizeof(float);
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&means_), ring_bytes));
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&tags_),
                          kRetrievalRingPages * sizeof(std::uint32_t)));
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&counts_),
                          kRetrievalRingPages * kRetrievalSubChunks * sizeof(std::uint32_t)));
    const std::size_t query_bytes =
        static_cast<std::size_t>(kRetrievalLayers) * kPageElems * sizeof(float);
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&query_), query_bytes));
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&frozen_query_), query_bytes));
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&last_query_), query_bytes));
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&query_rows_), sizeof(std::uint32_t)));
    // A page ring slot is "valid" only when its tag matches, so zeroing the tags is enough.
    KV_CHECK(cudaMemset(tags_, 0, kRetrievalRingPages * sizeof(std::uint32_t)));
    KV_CHECK(cudaMemset(means_, 0, ring_bytes));
    KV_CHECK(cudaMemset(query_, 0, query_bytes));
    KV_CHECK(cudaMemset(frozen_query_, 0, query_bytes));
    KV_CHECK(cudaMemset(last_query_, 0, query_bytes));
    KV_CHECK(cudaMemset(query_rows_, 0, sizeof(std::uint32_t)));
}

void KVRetrieval::begin_chunk(const std::int32_t* absolute_positions, std::int32_t tokens,
                              cudaStream_t stream) {
    if (tokens <= 0 || absolute_positions == nullptr) { return; }
    ensure_allocated();
    chunk_positions_.resize(static_cast<std::size_t>(tokens));
    if (chunk_positions_capacity_ < static_cast<std::size_t>(tokens)) {
        if (chunk_positions_device_ != nullptr) { (void)cudaFree(chunk_positions_device_); }
        KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&chunk_positions_device_),
                            static_cast<std::size_t>(tokens) * sizeof(std::int32_t)));
        chunk_positions_capacity_ = static_cast<std::size_t>(tokens);
    }
    // The position buffer is reused by every chunk and is read asynchronously by the capture kernels
    // of all sixteen layers, so the previous chunk has to be fully drained before it is overwritten.
    KV_CHECK(cudaStreamSynchronize(stream));
    KV_CHECK(cudaMemcpyAsync(chunk_positions_device_, absolute_positions,
                             static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                             cudaMemcpyDeviceToDevice, stream));
    // One small D2H per chunk: the page range has to be exact, and the alternative is a kernel
    // that rediscovers the boundaries on the device for every layer.
    KV_CHECK(cudaMemcpyAsync(chunk_positions_.data(), absolute_positions,
                               static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    KV_CHECK(cudaStreamSynchronize(stream));
    chunk_tokens_ = tokens;
    chunk_first_page_ =
        static_cast<std::uint32_t>(chunk_positions_.front()) / kRetrievalPageTokens;
    const std::uint32_t last_page =
        static_cast<std::uint32_t>(chunk_positions_.back()) / kRetrievalPageTokens;
    chunk_pages_.clear();
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        int probe_hits = -1;
        std::uint32_t probe_page = 0;
        if (const char* probe_env = std::getenv("NINFER_KV_PROBE_PAGE"); probe_env != nullptr) {
            probe_page = static_cast<std::uint32_t>(std::atoi(probe_env));
            probe_hits = 0;
            for (int t = 0; t < tokens; ++t) {
                if (static_cast<std::uint32_t>(chunk_positions_[static_cast<std::size_t>(t)]) /
                        kRetrievalPageTokens ==
                    probe_page) { ++probe_hits; }
            }
        }
        std::fprintf(stderr, "[kv-retrieve] chunk tokens=%d positions [%d,%d] pages [%u,%u] n=%u probe=%u hits=%d\n",
                     tokens, chunk_positions_.front(), chunk_positions_.back(), chunk_first_page_,
                     last_page, last_page - chunk_first_page_ + 1U, probe_page, probe_hits);
        std::fflush(stderr);
    }
    if (last_page < chunk_first_page_ || last_page - chunk_first_page_ + 1U > 4096U) {
        chunk_tokens_ = 0;
        return;
    }
    for (std::uint32_t page = chunk_first_page_; page <= last_page; ++page) {
        chunk_pages_.push_back(page);
    }
}

void KVRetrieval::accumulate_keys(const void* keys, int layer, cudaStream_t stream) {
    if (keys == nullptr || chunk_pages_.empty() || chunk_tokens_ <= 0) { return; }
    ensure_allocated();
    if (layer < 0 || layer >= kRetrievalLayers) { return; }
    kmeans_page_kernel<<<static_cast<unsigned>(chunk_pages_.size()), kPageElems, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(keys), layer, chunk_positions_device_, chunk_tokens_,
        chunk_first_page_, means_, tags_, counts_);
    if (layer == 0) {
        // A blocking device-to-host read after the first layer forces the ring writes for this chunk
        // to be visible before the window slides and the summaries get copied out. Without it the
        // archived pages come back all zero and every candidate ranks as "no data".
        std::uint32_t probe[8] = {};
        KV_CHECK(cudaMemcpy(probe, tags_, sizeof(probe), cudaMemcpyDeviceToHost));
        if (std::getenv("NINFER_KV_MEANS_TRACE") != nullptr) {
            std::fprintf(stderr, "[kv-tags] %u %u %u %u %u %u %u %u\n", probe[0], probe[1], probe[2],
                         probe[3], probe[4], probe[5], probe[6], probe[7]);
            std::fflush(stderr);
        }
    }
    KV_CHECK(cudaGetLastError());
}

void KVRetrieval::accumulate_query(const void* query, int layer, cudaStream_t stream) {
    if (query == nullptr || chunk_tokens_ <= 0) { return; }
    ensure_allocated();
    if (layer < 0 || layer >= kRetrievalLayers) { return; }
    kquery_page_kernel<<<1, kPageElems, 0, stream>>>(static_cast<const __nv_bfloat16*>(query),
                                                     layer, chunk_tokens_, query_);
    KV_CHECK(cudaGetLastError());
    // The question is the LAST token of the prompt, so keep a rolling copy of just that token's
    // query: summing over 98k positions dilutes it into the padding's mean direction.
    if (last_query_ != nullptr) {
        klast_query_kernel<<<1, kPageElems, 0, stream>>>(static_cast<const __nv_bfloat16*>(query),
                                                         chunk_tokens_, layer, last_query_);
        KV_CHECK(cudaGetLastError());
    }
}

void KVRetrieval::reset_query(cudaStream_t stream) {
    if (query_ == nullptr) { return; }
    KV_CHECK(cudaMemsetAsync(query_, 0,
                               static_cast<std::size_t>(kRetrievalLayers) * kPageElems *
                                   sizeof(float),
                               stream));
}

void KVRetrieval::freeze_query(cudaStream_t stream) {
    if (query_ == nullptr || frozen_query_ == nullptr) { return; }
    const float* source = last_query_ != nullptr ? last_query_ : query_;
    KV_CHECK(cudaMemcpyAsync(frozen_query_, source,
                             static_cast<std::size_t>(kRetrievalLayers) * kPageElems * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
    // The prompt is now complete: this snapshot is the question every archived page is scored
    // against, and only from here on may the window swap pages in.
    knormalize_query_kernel<<<kRetrievalLayers, kPageElems, kPageElems * sizeof(float), stream>>>(
        frozen_query_);
    prompt_complete_ = true;
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-retrieve] query frozen (last=%d)\n", last_query_ != nullptr ? 1 : 0);
        std::fflush(stderr);
    }
}

namespace {

// Host mirror of ops::normalized_hadamard_d256_inplace: lane l owns dims l+32*r, the five XOR
// stages run across lanes, then the three column stages run inside each lane, then 2^-4.
void hadamard_d256_host(const float* source, float* destination) {
    float lanes[32][8];
    for (int lane = 0; lane < 32; ++lane) {
        for (int r = 0; r < 8; ++r) { lanes[lane][r] = source[lane + 32 * r]; }
    }
    for (int stride = 1; stride <= 16; stride <<= 1) {
        float next[32][8];
        for (int lane = 0; lane < 32; ++lane) {
            const int peer_lane = lane ^ stride;
            for (int r = 0; r < 8; ++r) {
                const float value  = lanes[lane][r];
                const float peer   = lanes[peer_lane][r];
                next[lane][r]      = (lane & stride) == 0 ? value + peer : peer - value;
            }
        }
        std::memcpy(lanes, next, sizeof(lanes));
    }
    for (int lane = 0; lane < 32; ++lane) {
        for (int span = 1; span < 8; span <<= 1) {
            for (int base = 0; base < 8; base += 2 * span) {
                for (int offset = 0; offset < span; ++offset) {
                    const float low  = lanes[lane][base + offset];
                    const float high = lanes[lane][base + offset + span];
                    lanes[lane][base + offset]      = low + high;
                    lanes[lane][base + offset + span] = low - high;
                }
            }
        }
    }
    for (int lane = 0; lane < 32; ++lane) {
        for (int r = 0; r < 8; ++r) {
            destination[lane + 32 * r] = lanes[lane][r] * 0x1p-4f;
        }
    }
}

}  // namespace

float KVRetrieval::score_page_keys(const void* page_bytes, const float* query_rows) {
    if (page_bytes == nullptr || query_rows == nullptr) { return -1.0e30F; }
    const auto* page = static_cast<const std::byte*>(page_bytes);
    // Layout as the host page really is: per layer, K codes (32 KB), V codes (32 KB), K scales
    // (2 KB), V scales (4 KB) — 71,680 bytes per layer, 16 layers, page_stride 1,146,880.
    constexpr std::size_t kLayerStride      = 71680;
    constexpr std::size_t kKeyScaleInLayer  = 65536;
    constexpr std::size_t kKeyHeadStride    = 8192;
    constexpr std::size_t kKeyTokenStride   = 128;
    constexpr std::size_t kScaleHeadStride  = 512;
    constexpr std::size_t kScaleTokenStride = 8;
    const auto half_to_float = [](std::uint16_t bits) {
        const float sign   = (bits & 0x8000U) != 0U ? -1.0F : 1.0F;
        const int exponent = (bits >> 10) & 0x1FU;
        const int mantissa = bits & 0x3FFU;
        if (exponent == 0) { return sign * std::ldexp(static_cast<float>(mantissa), -24); }
        if (exponent == 31) { return sign * 65504.0F; }
        return sign * std::ldexp(static_cast<float>(mantissa + 1024), exponent - 25);
    };
    float best = -1.0e30F;
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(kRetrievalLayers); ++layer) {
        const std::byte* layer_codes  = page + layer * kLayerStride;
        const std::byte* layer_scales = page + layer * kLayerStride + kKeyScaleInLayer;
        for (std::size_t head = 0; head < static_cast<std::size_t>(kRetrievalKVHeads); ++head) {
            float rotated[256];
            hadamard_d256_host(query_rows + (layer * kRetrievalKVHeads + head) * 256U, rotated);
            for (std::size_t token = 0; token < 64; ++token) {
                const std::byte* codes = layer_codes + head * kKeyHeadStride + token * kKeyTokenStride;
                const std::byte* scale = layer_scales + head * kScaleHeadStride + token * kScaleTokenStride;
                float dot = 0.0F;
                for (std::size_t dim = 0; dim < 256; ++dim) {
                    std::uint16_t bits = 0;
                    std::memcpy(&bits, scale + 2 * (dim / 64), sizeof(bits));
                    const float group_scale = half_to_float(bits);
                    if (!(group_scale > 0.0F)) { dot = -1.0e30F; break; }
                    const std::uint8_t packed = static_cast<std::uint8_t>(codes[dim / 2]);
                    const int code = ((dim & 1U) == 0U) ? (static_cast<int>(packed & 0x0FU) - 8)
                                                        : (static_cast<int>((packed >> 4) & 0x0FU) - 8);
                    dot += rotated[dim] * static_cast<float>(code) * group_scale;
                }
                if (dot > best) { best = dot; }
            }
        }
    }
    return best;
}

void KVRetrieval::export_query(float* destination, cudaStream_t stream) const {
    if (destination == nullptr || frozen_query_ == nullptr) { return; }
    KV_CHECK(cudaMemcpyAsync(
        destination, frozen_query_,
        static_cast<std::size_t>(kRetrievalLayers) * kPageElems * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    KV_CHECK(cudaStreamSynchronize(stream));
}

void KVRetrieval::copy_page_means(std::uint32_t first_page, std::uint32_t page_count,
                                  void* destination, cudaStream_t stream) {
    if (destination == nullptr || page_count == 0) { return; }
    ensure_allocated();
    // Host-side loop over pages: a page count is in the tens, and this only happens when a block
    // leaves the window, so the round trips are not on any hot path.
    // The ring is filled by kernels that may run on a different stream than the one the archive is
    // using, so a stream-local barrier is not enough: a page archived while its capture kernel is
    // still in flight shows up as an empty tag, which silently drops that page from ranking.
    KV_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> staging(static_cast<std::size_t>(page_count) * kPageStride);
    std::vector<float> sums(static_cast<std::size_t>(kPageStride));
    std::uint32_t matched = 0;
    std::uint32_t live_count = 0;
    for (std::uint32_t index = 0; index < page_count; ++index) {
        const std::uint32_t page = first_page + index;
        const std::uint32_t slot = page % static_cast<std::uint32_t>(kRetrievalRingPages);
        std::uint32_t tag = 0;
        KV_CHECK(cudaMemcpy(&tag, tags_ + slot, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        if (index == 0) { live_count = tag; }
        if (std::getenv("NINFER_KV_MEANS_TRACE") != nullptr && index < 12) {
            std::fprintf(stderr, "[kv-means] page=%u slot=%zu tag=%u\n", page, slot, tag);
        }
        if (tag != page + 1U) { continue; }
        ++matched;
        KV_CHECK(cudaMemcpy(sums.data(), means_ + static_cast<std::size_t>(slot) * kPageStride,
                              static_cast<std::size_t>(kPageStride) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        if (const char* probe_env = std::getenv("NINFER_KV_PROBE_PAGE"); probe_env != nullptr &&
            page == static_cast<std::uint32_t>(std::atoi(probe_env))) {
            std::fprintf(stderr, "[kv-page] page=%u sum=%.5f lo=%.5f hi=%.5f\n", page, sums[0],
                         sums[kPageHalf], sums[2 * kPageHalf]);
            std::fflush(stderr);
        }
        // Quest-style bound: keep the per-channel extent of the page's keys instead of their mean,
        // so a page that merely resembles the corpus average can no longer outrank the page that
        // actually contains the needle.
        for (int layer = 0; layer < kRetrievalLayers; ++layer) {
            const std::size_t base = static_cast<std::size_t>(layer) * kPageElems;
            const std::size_t dest = static_cast<std::size_t>(index) * kPageStride + base;
            std::uint32_t sub_counts = 0;
            KV_CHECK(cudaMemcpy(&sub_counts, counts_ + slot * kRetrievalSubChunks,
                                sizeof(sub_counts), cudaMemcpyDeviceToHost));
            const float inverse = sub_counts == 0U ? 1.0F : 1.0F / static_cast<float>(sub_counts);
            float norm_squared = 0.0F;
            for (int element = 0; element < kPageElems; ++element) {
                const float mean = sums[base + element] * inverse;
                norm_squared += mean * mean;
            }
            const float scale = 1.0F / (std::sqrt(norm_squared) + 1.0e-6F);
            for (int element = 0; element < kPageElems; ++element) {
                staging[dest + element] = __float2bfloat16(sums[base + element] * inverse * scale);
                staging[dest + kPageHalf + element] =
                    __float2bfloat16(sums[kPageHalf + base + element] * scale);
                staging[dest + 2 * kPageHalf + element] =
                    __float2bfloat16(sums[2 * kPageHalf + base + element] * scale);
            }
        }
    }
    std::memcpy(destination, staging.data(),
                static_cast<std::size_t>(page_count) * kPageStride * sizeof(__nv_bfloat16));
    if (const char* probe_env = std::getenv("NINFER_KV_PROBE_PAGE"); probe_env != nullptr) {
        const std::uint32_t probe = static_cast<std::uint32_t>(std::atoi(probe_env));
        if (probe >= first_page && probe < first_page + page_count) {
            const auto* back = reinterpret_cast<const __nv_bfloat16*>(
                static_cast<const char*>(destination) +
                static_cast<std::size_t>(probe - first_page) * kRetrievalPageMeanBytes);
            std::fprintf(stderr, "[kv-h2d] page=%u index=%u buf=%p back=%.6f %.6f %.6f %.6f\n", probe,
                         probe - first_page, destination, static_cast<double>(__bfloat162float(back[0])),
                         static_cast<double>(__bfloat162float(back[1])),
                         static_cast<double>(__bfloat162float(back[2])),
                         static_cast<double>(__bfloat162float(back[3])));
            std::fflush(stderr);
        }
    }
    int nonzero = 0;
    for (std::uint32_t page_index = 0; page_index < page_count; ++page_index) {
        const auto* written = reinterpret_cast<const __nv_bfloat16*>(
            static_cast<const char*>(destination) +
            static_cast<std::size_t>(page_index) * kRetrievalPageMeanBytes);
        if (__bfloat162float(written[0]) != 0.0F || __bfloat162float(written[1]) != 0.0F ||
            __bfloat162float(written[2]) != 0.0F) { ++nonzero; }
    }
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-write] pages [%u,%u) matched=%u nonzero=%u dest=%p\n", first_page,
                     first_page + page_count, matched, nonzero, destination);
        std::fflush(stderr);
    }
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-retrieve] copied means for pages [%u,%u) matched=%u first_slot_tag=%u\n",
                     first_page, first_page + page_count, matched, live_count);
        std::fflush(stderr);
    }
}

void KVRetrieval::forget(std::uint32_t absolute_page, cudaStream_t stream) {
    (void)absolute_page;
    (void)stream;
    // Ring slots are validated by their tag, so a page that leaves without being archived needs
    // no explicit release: the tag simply stops matching once the slot is reused.
}

namespace {
// IEEE-754 binary16 to float, host side: the archived scales are plain fp16 and this file's debug
// path cannot use the device-only __half type.
float decode_fp16(std::uint16_t bits) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits >> 15) & 0x1U;
    const std::uint32_t exponent = static_cast<std::uint32_t>(bits >> 10) & 0x1FU;
    const std::uint32_t mantissa = static_cast<std::uint32_t>(bits) & 0x3FFU;
    float value                  = 0.0F;
    if (exponent == 0U) {
        value = static_cast<float>(mantissa) * 5.9604644775390625e-08F;
    } else if (exponent == 0x1FU) {
        value = mantissa == 0U ? std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::quiet_NaN();
    } else {
        const float power = std::ldexp(1.0F, static_cast<int>(exponent) - 15);
        value             = power * (1.0F + static_cast<float>(mantissa) / 1024.0F);
    }
    return sign != 0U ? -value : value;
}
} // namespace

void KVRetrieval::debug_scan_page_keys(const void* page_bytes, std::uint32_t absolute_page) {
    if (page_bytes == nullptr) { return; }
    const auto* bytes = static_cast<const std::uint8_t*>(page_bytes);
    double code_sum   = 0.0;
    std::uint32_t count = 0;
    std::uint32_t nibble_hist[16] = {};
    for (std::uint32_t index = 0; index < 2048U; ++index) {
        for (int half = 0; half < 2; ++half) {
            const std::uint32_t nibble = (bytes[index] >> (4 * half)) & 0x0FU;
            ++nibble_hist[nibble];
            code_sum += static_cast<double>(static_cast<int>(nibble ^ 8U) - 8);
            ++count;
        }
    }
    std::fprintf(stderr, "[kv-deq] page=%u codes=%u mean_code=%.3f hist=[", absolute_page, count,
                 count != 0 ? code_sum / static_cast<double>(count) : 0.0);
    for (std::uint32_t nibble = 0; nibble < 16U; ++nibble) {
        std::fprintf(stderr, "%u%s", nibble_hist[nibble], nibble + 1U == 16U ? "" : " ");
    }
    std::fprintf(stderr, "]\n");
    std::fflush(stderr);
    // Decode one 256-dim key row at a caller-chosen byte offset: the row is 128 packed codes plus
    // four fp16 G64 scales. A real key row decodes to near-zero mean with a small spread, which is
    // how the exact layer/head/token offsets get confirmed without guessing the plane layout.
    if (std::getenv("NINFER_KV_DEQ_SCAN") != nullptr) {
        const std::size_t limit = 1U << 20;
        std::size_t found       = 0;
        for (std::size_t offset = 0; offset + 8U < limit && found < 6U; offset += 8U) {
            float decoded[4] = {};
            bool  plausible  = true;
            for (int slot = 0; slot < 4; ++slot) {
                std::uint16_t bits = 0;
                std::memcpy(&bits, bytes + offset + 2U * static_cast<std::size_t>(slot),
                            sizeof(bits));
                decoded[slot] = decode_fp16(bits);
                if (!(decoded[slot] > 1.0e-4F && decoded[slot] < 2.0e-1F)) { plausible = false; }
            }
            if (!plausible) { continue; }
            // A real scale plane is a long contiguous run of plausible tuples; scattered hits are
            // just code bytes that happen to decode into the range, so require a run of 32.
            std::size_t run = 0;
            for (std::size_t probe = offset; probe + 8U < limit; probe += 8U) {
                float probe_values[4] = {};
                bool  probe_ok        = true;
                for (int slot = 0; slot < 4; ++slot) {
                    std::uint16_t bits = 0;
                    std::memcpy(&bits, bytes + probe + 2U * static_cast<std::size_t>(slot),
                                sizeof(bits));
                    probe_values[slot] = decode_fp16(bits);
                    if (!(probe_values[slot] > 1.0e-4F && probe_values[slot] < 2.0e-1F)) {
                        probe_ok = false;
                    }
                }
                if (!probe_ok) { break; }
                ++run;
            }
            if (run < 32U) { continue; }
            std::fprintf(stderr, "[kv-deq-scale] off=%zu s=%.5f %.5f %.5f %.5f\n", offset,
                         static_cast<double>(decoded[0]), static_cast<double>(decoded[1]),
                         static_cast<double>(decoded[2]), static_cast<double>(decoded[3]));
            ++found;
        }
        std::fflush(stderr);
    }
    const char* offset_env = std::getenv("NINFER_KV_DEQ_OFFSET");    if (offset_env == nullptr) { return; }
    const std::size_t offset = static_cast<std::size_t>(std::strtoul(offset_env, nullptr, 10));
    // Two orderings fit the derived 280-byte record: [K codes][K scales][V codes][V scales] puts the
    // K scales at +128, while [K codes][V codes][K scales][V scales] puts them at +256. Print both so
    // the plausible one (positive scales, sane spread) can be picked from data instead of guessed.
    for (const std::size_t scale_offset : {32768U, 524288U, 1048576U}) {
        const auto* candidate = reinterpret_cast<const std::uint16_t*>(bytes + offset + scale_offset);
        float candidate_values[4] = {};
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            candidate_values[slot] = decode_fp16(candidate[slot]);
        }
        float variance_sum = 0.0F;
        float sum_value    = 0.0F;
        std::size_t counted = 0;
        for (std::size_t dim = 0; dim < 256U; ++dim) {
            const float scale = candidate_values[dim / 64U];
            if (scale <= 0.0F) { counted = 0; break; }
            ++counted;
        }
        (void)variance_sum;
        (void)sum_value;
        std::fprintf(stderr, "[kv-deq-try] off=%zu scale_at=%zu s=%.5f %.5f %.5f %.5f valid=%zu\n",
                     offset, offset + scale_offset, static_cast<double>(candidate_values[0]),
                     static_cast<double>(candidate_values[1]),
                     static_cast<double>(candidate_values[2]),
                     static_cast<double>(candidate_values[3]), counted);
    }
    std::fflush(stderr);
    const char* scale_env = std::getenv("NINFER_KV_DEQ_SCALE");
    const std::size_t scale_offset_choice =
        scale_env != nullptr ? static_cast<std::size_t>(std::strtoul(scale_env, nullptr, 10)) : 128U;
    const auto* scales = reinterpret_cast<const std::uint16_t*>(bytes + offset + scale_offset_choice);
    float low = 1.0e30F, high = -1.0e30F, sum = 0.0F, sum_sq = 0.0F;
    for (std::size_t dim = 0; dim < 256U; ++dim) {
        const float scale = decode_fp16(scales[dim / 64U]);
        const std::uint8_t code_byte = bytes[offset + dim / 2U];
        const int code = ((dim & 1U) == 0U) ? (static_cast<int>(code_byte & 0x0FU) - 8)
                                           : (static_cast<int>((code_byte >> 4) & 0x0FU) - 8);
        const float value = static_cast<float>(code) * scale;
        low               = fminf(low, value);
        high              = fmaxf(high, value);
        sum += value;
        sum_sq += value * value;
    }
    const float mean     = sum / 256.0F;
    const float variance = fmaxf(0.0F, sum_sq / 256.0F - mean * mean);
    std::fprintf(stderr, "[kv-deq-row] off=%zu min=%.5f max=%.5f mean=%.5f std=%.5f s0=%.5f s1=%.5f\n",
                 offset, static_cast<double>(low), static_cast<double>(high),
                 static_cast<double>(mean), static_cast<double>(sqrtf(variance)),
                 static_cast<double>(decode_fp16(scales[0])),
                 static_cast<double>(decode_fp16(scales[1])));
    std::fflush(stderr);
}

std::vector<RetrievalCandidate> KVRetrieval::rank(const void* host_page_means,
                                                 const std::vector<std::uint32_t>& absolute_pages,
                                                 cudaStream_t stream) {
    if (host_page_means == nullptr || absolute_pages.empty()) { return {}; }
    ensure_allocated();
    const int entries = static_cast<int>(absolute_pages.size());
    __nv_bfloat16* device_pointers = nullptr;
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_pointers),
                        static_cast<std::size_t>(entries) * kPageStride * sizeof(__nv_bfloat16)));
    KV_CHECK(cudaMemcpy(device_pointers, host_page_means,
                        static_cast<std::size_t>(entries) * kPageStride * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
    float* device_scores = nullptr;
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_scores),
                        sizeof(float) * static_cast<std::size_t>(entries) * kScoreUnits));
    std::vector<float> raw(static_cast<std::size_t>(entries) * kScoreUnits, 0.0F);
    std::vector<float> scores(static_cast<std::size_t>(entries), 0.0F);
    const int threads = kPageElems;
    const int shared  = threads * static_cast<int>(sizeof(float));
    float* device_mean = nullptr;
    // The per-block summaries were handed over with cudaMemcpyAsync from a host staging vector;
    // scoring must not start before those copies have landed.
    KV_CHECK(cudaStreamSynchronize(stream));
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_mean), sizeof(float) * kPageStride));
    kmean_page_kernel<<<1, 256, 0, stream>>>(device_pointers, entries, device_mean);
    KV_CHECK(cudaGetLastError());
    KV_CHECK(cudaMemsetAsync(device_scores, 0xFF, sizeof(float) * entries * kScoreUnits, stream));
    {
        const char* bound_env  = std::getenv("NINFER_KV_BOUND");
        const int   use_bound = bound_env != nullptr && bound_env[0] != '0' ? 1 : 0;
        KV_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_kv_use_bound), &use_bound, sizeof(use_bound)));
    }
    kscore_page_kernel<<<static_cast<unsigned>(entries), threads, shared, stream>>>(
        device_pointers, frozen_query_ != nullptr ? frozen_query_ : query_, device_mean,
        device_scores, entries);
    KV_CHECK(cudaGetLastError());
    KV_CHECK(cudaMemcpyAsync(raw.data(), device_scores,
                             sizeof(float) * static_cast<std::size_t>(entries) * kScoreUnits,
                             cudaMemcpyDeviceToHost, stream));
    KV_CHECK(cudaStreamSynchronize(stream));
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::size_t nonzero = 0;
        for (int index = 0; index < entries; ++index) {
            float peak = 0.0F;
            for (int unit = 0; unit < kScoreUnits; ++unit) {
                peak = std::max(
                    peak, std::fabs(raw[static_cast<std::size_t>(index) * kScoreUnits + unit]));
            }
            if (peak > 0.0F) { ++nonzero; }
        }
        std::fprintf(stderr, "[kv-rank] nonzero=%zu/%d\n", nonzero, entries);
        std::fflush(stderr);
    }
    KV_CHECK(cudaFree(device_scores));
    KV_CHECK(cudaFree(device_mean));
    KV_CHECK(cudaFree(device_pointers));

    // Every layer votes with the same weight: raw cosines are not comparable between layers, so
    // each layer's scores are standardised across the candidates before they are averaged. Without
    // this the layer with the widest dynamic range decides the ranking on its own.
    std::vector<float> unit_sums(static_cast<std::size_t>(entries), 0.0F);
    for (int unit = 0; unit < kScoreUnits; ++unit) {
        float mean = 0.0F;
        for (int index = 0; index < entries; ++index) {
            mean += raw[static_cast<std::size_t>(index) * kScoreUnits + unit];
        }
        mean /= static_cast<float>(entries);
        float variance = 0.0F;
        for (int index = 0; index < entries; ++index) {
            const float delta = raw[static_cast<std::size_t>(index) * kScoreUnits + unit] - mean;
            variance += delta * delta;
        }
        const float deviation = std::sqrt(variance / static_cast<float>(entries)) + 1.0e-9F;
        for (int index = 0; index < entries; ++index) {
            const float value = raw[static_cast<std::size_t>(index) * kScoreUnits + unit];
            // An unfilled unit keeps the kernel's -1e30 sentinel; letting it through drags one page
            // far below every other and poisons the averaged score.
            // A unit that is constant across candidates carries no signal: its deviation collapses
            // to the epsilon and the division would otherwise explode one page to -1e30.
            const float z = (value <= -1.0e29F || deviation < 1.0e-6F)
                                ? 0.0F
                                : (value - mean) / deviation;
            unit_sums[static_cast<std::size_t>(index)] += z;
            float& slot   = scores[static_cast<std::size_t>(index)];
            if (unit == 0 || z > slot) { slot = z; }
        }
    }
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-unit] top page per (layer,head):");
        for (int unit = 0; unit < kScoreUnits; ++unit) {
            int best = 0;
            for (int index = 1; index < entries; ++index) {
                if (raw[static_cast<std::size_t>(index) * kScoreUnits + unit] >
                    raw[static_cast<std::size_t>(best) * kScoreUnits + unit]) {
                    best = index;
                }
            }
            std::fprintf(stderr, " %d:%u", unit, absolute_pages[static_cast<std::size_t>(best)]);
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }

    if (const char* agg = std::getenv("NINFER_KV_AGG"); agg == nullptr || agg[0] != 'x') {
        for (int index = 0; index < entries; ++index) {
            scores[static_cast<std::size_t>(index)] =
                unit_sums[static_cast<std::size_t>(index)] / static_cast<float>(kScoreUnits);
        }
    }
    // A page whose summary never landed in the ring keeps an all-zero record. Scoring it returns the
    // same constant for every such page, which used to tie a large block of candidates at the top and
    // push the real best page out of the returned prefix. Drop them instead of ranking them.
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        int untouched = 0;
        for (const float value : raw) {
            if (!std::isfinite(value)) { ++untouched; }
        }
        std::fprintf(stderr, "[kv-score] untouched units=%d/%d\n", untouched, entries * kScoreUnits);
        std::fflush(stderr);
    }
    int live_candidates = 0;
    for (int index = 0; index < entries; ++index) {
        bool any = false;
        for (int unit = 0; unit < kScoreUnits && !any; ++unit) {
            any = raw[static_cast<std::size_t>(index) * kScoreUnits + unit] != 0.0F;
        }
        if (any) {
            ++live_candidates;
        } else {
            scores[static_cast<std::size_t>(index)] = -1.0e30F;
        }
    }
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-score] live candidates=%d/%d\n", live_candidates, entries);
        std::fflush(stderr);
    }
    if (const char* detrend = std::getenv("NINFER_KV_DETREND"); detrend != nullptr && detrend[0] == '1') {
        std::vector<float> trend(static_cast<std::size_t>(entries), 0.0F);
        constexpr std::uint32_t kBand = 64;
        for (int index = 0; index < entries; ++index) {
            const std::uint32_t page = absolute_pages[static_cast<std::size_t>(index)];
            float sum = 0.0F;
            int count = 0;
            for (int other = 0; other < entries; ++other) {
                const std::uint32_t other_page = absolute_pages[static_cast<std::size_t>(other)];
                const std::uint32_t distance = other_page > page ? other_page - page : page - other_page;
                if (distance <= kBand) {
                    sum += scores[static_cast<std::size_t>(other)];
                    ++count;
                }
            }
            trend[static_cast<std::size_t>(index)] =
                count > 0 ? sum / static_cast<float>(count) : 0.0F;
        }
        for (int index = 0; index < entries; ++index) {
            scores[static_cast<std::size_t>(index)] -= trend[static_cast<std::size_t>(index)];
        }
    }

    // Only the recall that runs once prefill is complete decides whether the needle is reachable;
    // later decode-time rounds see a different candidate set and would quietly rewrite the metric.
    static int probe_round = 0;
    ++probe_round;
    if (const char* probe_env = std::getenv("NINFER_KV_PROBE_PAGE");
        probe_env != nullptr && probe_round == 1) {
        const std::uint32_t probe = static_cast<std::uint32_t>(std::atoi(probe_env));
        int rank = 0;
        float probe_score = 0.0F;
        int found = 0;
        int probe_index = 0;
        for (int index = 0; index < entries; ++index) {
            if (absolute_pages[static_cast<std::size_t>(index)] == probe) {
                probe_score = scores[static_cast<std::size_t>(index)];
                probe_index = index;
                found = 1;
                for (int other = 0; other < entries; ++other) {
                    if (scores[static_cast<std::size_t>(other)] > probe_score) { ++rank; }
                }
                break;
            }
        }
        if (found != 0) {
            const std::size_t at = static_cast<std::size_t>(probe_index) * kScoreUnits;
            std::fprintf(stderr, "[kv-raw] page=%u u0=%.4f u1=%.4f u2=%.4f u63=%.4f\n", probe,
                         raw[at], raw[at + 1], raw[at + 2], raw[at + kScoreUnits - 1]);
        }
        std::fprintf(stderr, "[kv-probe] page=%u found=%d rank=%d z=%.6f\n", probe, found, rank,
                     probe_score);
        std::fflush(stderr);
    }

    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        int bad = 0;
        float lo = 1.0e30F;
        float hi = -1.0e30F;
        for (const float value : scores) {
            if (!std::isfinite(value)) { ++bad; continue; }
            lo = std::min(lo, value);
            hi = std::max(hi, value);
        }
        std::fprintf(stderr, "[kv-score] entries=%zu nonfinite=%d min=%.6f max=%.6f\n",
                     scores.size(), bad, static_cast<double>(lo), static_cast<double>(hi));
        std::fflush(stderr);
    }

    // Softmax over the candidates, then best-first. Scores from different page pairs are not
    // comparable across calls, but that is all the caller needs: a ranking, not a threshold.
    float peak = scores.front();
    for (const float value : scores) { peak = std::max(peak, value); }
    float total = 0.0F;
    for (float& value : scores) {
        value = std::exp(value - peak);
        total += value;
    }
    std::vector<RetrievalCandidate> ranked;
    ranked.reserve(scores.size());
    for (std::size_t index = 0; index < scores.size(); ++index) {
        ranked.push_back({absolute_pages[index], total > 0.0F ? scores[index] / total : 0.0F});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const RetrievalCandidate& a, const RetrievalCandidate& b) {
                  if (a.score != b.score) { return a.score > b.score; }
                  return a.absolute_page < b.absolute_page;
              });
    return ranked;
}

}  // namespace ninfer::models::qwen3_5::detail
