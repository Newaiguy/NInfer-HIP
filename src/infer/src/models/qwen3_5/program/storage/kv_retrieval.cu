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
constexpr int kPageStride  = kRetrievalLayers * kLayerStride;        // per-page summary
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
        fresh = (tags[slot] != page) ? 1 : 0;
        if (fresh != 0) {
            tags[slot] = page;
            for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
                counts[slot * kRetrievalSubChunks + sub] = 0;
            }
        }
    }
    __syncthreads();
    if (tid < kPageElems) {
        float acc[kRetrievalSubChunks];
        for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
            const std::size_t off = static_cast<std::size_t>(slot) * kPageStride +
                                    static_cast<std::size_t>(sub) * kLayerStride +
                                    static_cast<std::size_t>(layer) * kPageElems + tid;
            acc[sub] = (fresh != 0) ? 0.0F : ring[off];
        }
        for (int t = 0; t < tokens; ++t) {
            const std::uint32_t pos = static_cast<std::uint32_t>(positions[t]);
            if (pos / kRetrievalPageTokens != page) { continue; }
            const int sub = static_cast<int>((pos % kRetrievalPageTokens) / kRetrievalSubTokens);
            acc[sub] += __bfloat162float(keys[static_cast<std::size_t>(t) * kPageElems + tid]);
        }
        for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
            const std::size_t off = static_cast<std::size_t>(slot) * kPageStride +
                                    static_cast<std::size_t>(sub) * kLayerStride +
                                    static_cast<std::size_t>(layer) * kPageElems + tid;
            ring[off] = acc[sub];
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
__global__ void kscore_page_kernel(const __nv_bfloat16* const* __restrict__ page_means,
                                   const float* __restrict__ query, const float* __restrict__ mean,
                                   float* __restrict__ scores, int entries) {
    extern __shared__ float shared[];
    const int entry = blockIdx.x;
    if (entry >= entries) { return; }
    const __nv_bfloat16* page = page_means[entry];
    const int tid             = threadIdx.x;
    for (int layer = 0; layer < kRetrievalLayers; ++layer) {
        for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
            float local = 0.0F;
            if (tid < kPageElems) {
                const std::size_t offset = static_cast<std::size_t>(sub) * kLayerStride +
                                           static_cast<std::size_t>(layer) * kPageElems + tid;
                local = query[static_cast<std::size_t>(layer) * kPageElems + tid] *
                        (__bfloat162float(page[offset]) - mean[offset]);
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
__global__ void kmean_page_kernel(const __nv_bfloat16* const* __restrict__ page_means, int entries,
                                  float* __restrict__ mean) {
    for (int idx = threadIdx.x; idx < kPageStride; idx += blockDim.x) {
        float sum = 0.0F;
        for (int entry = 0; entry < entries; ++entry) {
            sum += __bfloat162float(page_means[entry][idx]);
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
        std::fprintf(stderr, "[kv-retrieve] chunk tokens=%d positions [%d,%d] pages [%u,%u] n=%u\n",
                     tokens, chunk_positions_.front(), chunk_positions_.back(), chunk_first_page_,
                     last_page, last_page - chunk_first_page_ + 1U);
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

void KVRetrieval::copy_page_means(std::uint32_t first_page, std::uint32_t page_count,
                                  void* destination, cudaStream_t stream) {
    if (destination == nullptr || page_count == 0) { return; }
    ensure_allocated();
    // Host-side loop over pages: a page count is in the tens, and this only happens when a block
    // leaves the window, so the round trips are not on any hot path.
    std::vector<__nv_bfloat16> staging(static_cast<std::size_t>(page_count) * kPageStride);
    std::vector<float> sums(static_cast<std::size_t>(kPageStride));
    for (std::uint32_t index = 0; index < page_count; ++index) {
        const std::uint32_t page = first_page + index;
        const std::uint32_t slot = page % static_cast<std::uint32_t>(kRetrievalRingPages);
        std::uint32_t tag = 0;
        KV_CHECK(cudaMemcpy(&tag, tags_ + slot, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        if (tag != page) { continue; }
        std::uint32_t sub_counts[kRetrievalSubChunks];
        KV_CHECK(cudaMemcpy(sub_counts, counts_ + slot * kRetrievalSubChunks,
                              sizeof(sub_counts), cudaMemcpyDeviceToHost));
        KV_CHECK(cudaMemcpy(sums.data(), means_ + static_cast<std::size_t>(slot) * kPageStride,
                              static_cast<std::size_t>(kPageStride) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // Per-(sub-chunk, layer) unit norm: the raw mean's magnitude is set by how peaked the
        // page's keys are, so an unnormalised dot lets one layer drown every other one.
        for (int sub = 0; sub < kRetrievalSubChunks; ++sub) {
            if (sub_counts[sub] == 0) { continue; }
            const float inverse = 1.0F / static_cast<float>(sub_counts[sub]);
            for (int layer = 0; layer < kRetrievalLayers; ++layer) {
                const std::size_t base = static_cast<std::size_t>(sub) * kLayerStride +
                                         static_cast<std::size_t>(layer) * kPageElems;
                float norm_squared = 0.0F;
                for (int element = 0; element < kPageElems; ++element) {
                    const float value = sums[base + element] * inverse;
                    norm_squared += value * value;
                }
                const float scale = 1.0F / (std::sqrt(norm_squared) + 1.0e-6F);
                for (int element = 0; element < kPageElems; ++element) {
                    staging[static_cast<std::size_t>(index) * kPageStride + base + element] =
                        __float2bfloat16(sums[base + element] * inverse * scale);
                }
            }
        }
    }
    KV_CHECK(cudaMemcpyAsync(destination, staging.data(),
                               static_cast<std::size_t>(page_count) * kPageStride *
                                   sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    if (std::getenv("NINFER_KV_RETRIEVE_TRACE") != nullptr) {
        std::fprintf(stderr, "[kv-retrieve] copied means for pages [%u,%u)\n", first_page,
                     first_page + page_count);
        std::fflush(stderr);
    }
}

void KVRetrieval::forget(std::uint32_t absolute_page, cudaStream_t stream) {
    (void)absolute_page;
    (void)stream;
    // Ring slots are validated by their tag, so a page that leaves without being archived needs
    // no explicit release: the tag simply stops matching once the slot is reused.
}

std::vector<RetrievalCandidate> KVRetrieval::rank(const std::vector<const void*>& page_means,
                                                 const std::vector<std::uint32_t>& absolute_pages,
                                                 cudaStream_t stream) {
    if (page_means.empty() || page_means.size() != absolute_pages.size()) { return {}; }
    ensure_allocated();
    const int entries = static_cast<int>(page_means.size());
    std::vector<const __nv_bfloat16*> pointers(static_cast<std::size_t>(entries), nullptr);
    for (int index = 0; index < entries; ++index) {
        pointers[static_cast<std::size_t>(index)] =
            static_cast<const __nv_bfloat16*>(page_means[static_cast<std::size_t>(index)]);
    }
    const __nv_bfloat16** device_pointers = nullptr;
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_pointers),
                        sizeof(const __nv_bfloat16*) * static_cast<std::size_t>(entries)));
    KV_CHECK(cudaMemcpyAsync(device_pointers, pointers.data(),
                             sizeof(const __nv_bfloat16*) * static_cast<std::size_t>(entries),
                             cudaMemcpyHostToDevice, stream));
    float* device_scores = nullptr;
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_scores),
                        sizeof(float) * static_cast<std::size_t>(entries) * kScoreUnits));
    std::vector<float> raw(static_cast<std::size_t>(entries) * kScoreUnits, 0.0F);
    std::vector<float> scores(static_cast<std::size_t>(entries), 0.0F);
    const int threads = kPageElems;
    const int shared  = threads * static_cast<int>(sizeof(float));
    float* device_mean = nullptr;
    KV_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_mean), sizeof(float) * kPageStride));
    kmean_page_kernel<<<1, 256, 0, stream>>>(device_pointers, entries, device_mean);
    KV_CHECK(cudaGetLastError());
    kscore_page_kernel<<<static_cast<unsigned>(entries), threads, shared, stream>>>(
        device_pointers, frozen_query_ != nullptr ? frozen_query_ : query_, device_mean,
        device_scores, entries);
    KV_CHECK(cudaGetLastError());
    KV_CHECK(cudaMemcpyAsync(raw.data(), device_scores,
                             sizeof(float) * static_cast<std::size_t>(entries) * kScoreUnits,
                             cudaMemcpyDeviceToHost, stream));
    KV_CHECK(cudaStreamSynchronize(stream));
    KV_CHECK(cudaFree(device_scores));
    KV_CHECK(cudaFree(device_mean));
    KV_CHECK(cudaFree(const_cast<__nv_bfloat16**>(device_pointers)));

    // Every layer votes with the same weight: raw cosines are not comparable between layers, so
    // each layer's scores are standardised across the candidates before they are averaged. Without
    // this the layer with the widest dynamic range decides the ranking on its own.
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
            const float z = (raw[static_cast<std::size_t>(index) * kScoreUnits + unit] - mean) / deviation;
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
                  return a.score > b.score;
              });
    return ranked;
}

}  // namespace ninfer::models::qwen3_5::detail
