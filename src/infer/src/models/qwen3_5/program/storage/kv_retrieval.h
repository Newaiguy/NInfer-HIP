#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Leg B / kvmem W4: retrieval scoring on pre-RoPE keys.
//
// The archived pages hold K rows that were already rotated with their true absolute
// positions, which is what makes a recalled page score correctly at any slot -- but it also
// means the archive cannot be compared against a future query, because the rotation differs
// per token. Retrieval therefore keeps a separate, position-free summary of each page: the
// MEAN of its pre-RoPE K rows, captured where the rotation happens (text.cpp, between the key
// RMSNorm and the RoPE call). A query is turned into the same kind of summary by averaging its
// pre-RoPE Q rows, and a page scores as the dot product of the two.
//
// Only the 16 full-attention layers have a KV cache in this model (the other 48 layers are
// linear attention with a constant-size state), so a page summary is
//   kRetrievalLayers * kRetrievalKVHeads * kRetrievalHeadDim * 2 bytes = 32 KB.
// The ring below only has to cover the resident window (320 pages at a 20480-token window),
// not the whole context: a page's summary is handed to the archive when the page is archived,
// and the ring slot is free to be reused after that.
inline constexpr int kRetrievalLayers     = 16;
inline constexpr int kRetrievalKVHeads    = 4;
inline constexpr int kRetrievalQueryHeads = 24;
inline constexpr int kRetrievalHeadDim    = 256;
inline constexpr int kRetrievalPageTokens = 64;
// Must cover the whole resident window: a page's summary is captured while it is resident, and
// a page that finds no free slot simply never gets scored (its archived copy stays zero, which
// shows up as "every candidate has the same score" = 1/candidates). 1152 pages = a 65536-token
// window plus the sink and some slack; the buffer costs 256 KB per slot on the device.
inline constexpr int kRetrievalRingPages  = 2048;

// A single mean per page dilutes a one-sentence needle to 1/64 of the summary, and the ranking
// then measures "how typical is this page" instead of "how relevant". Each page is therefore
// summarized at kRetrievalSubChunks resolutions and a page scores as its BEST sub-chunk.
inline constexpr int kRetrievalSubChunks = 1;
inline constexpr int kRetrievalSubTokens = kRetrievalPageTokens / kRetrievalSubChunks;

// Bytes of one page's summary record:
// 4 sub-chunks x 16 layers x 4 kv heads x 256 dims of bf16.
inline constexpr std::size_t kRetrievalPageMeanBytes =
    3U * static_cast<std::size_t>(kRetrievalLayers) * kRetrievalKVHeads * kRetrievalHeadDim *
    sizeof(std::uint16_t);

// Set when recall_into_tail() actually moved archived pages into the window. The last prefill
// chunk is replayed only in that case: the replay exists to let the newest tokens attend to the
// recalled pages, so without a recall it would just re-run work -- and at a window that never
// slid it re-runs it in the wrong coordinate system.
inline bool g_kv_recall_loaded = false;

struct RetrievalCandidate {
    std::uint32_t absolute_page;
    float score;
};

class KVRetrieval {
public:
    // Enabled by NINFER_KV_RETRIEVE=1. Everything stays off (no allocation, no per-layer work)
    // when it is not set, so the default path is untouched.
    static bool enabled();
    static KVRetrieval* instance() noexcept;

    // Called once per forward chunk, before the layer loop. Copies the chunk's absolute
    // positions to the host so the per-layer accumulation can hand the kernel an exact page
    // range instead of guessing one; a chunk spans at most ceil(tokens/64)+1 pages.
    void begin_chunk(const std::int32_t* absolute_positions, std::int32_t tokens,
                     cudaStream_t stream);

    // Per-layer: fold this chunk's pre-RoPE key rows into the page summaries.
    void accumulate_keys(const void* keys, int layer, cudaStream_t stream);

    // Per-layer: fold the chunk's pre-RoPE query rows into the query summary. Only the trailing
    // tokens matter (the query is "what am I looking for now"), so callers pass the whole chunk
    // and the last chunk before a slide carries the newest window.
    void accumulate_query(const void* query, int layer, cudaStream_t stream);

    void reset_query(cudaStream_t stream);

    // Snapshots the current query summary. Call it when the prompt is complete: the ranking done at
    // the first decode step must use the question, not the chunk that happens to be in flight then.
    void freeze_query(cudaStream_t stream);

    // True once the prompt is complete and freeze_query() has snapshotted the question.
    [[nodiscard]] bool prompt_complete() const noexcept { return prompt_complete_; }

    // Copies the frozen query rows to `destination` laid out as [layer][kv_head][dim] floats. The
    // two-stage rerank decodes archived keys on the host, so it needs the query on the host too.
    void export_query(float* destination, cudaStream_t stream) const;

    // Two-stage rerank: decode one archived page's int4 keys and contract them with the rotated
    // query rows ([layer][kv_head][dim] as exported above). Mirrors the device path exactly — the
    // stored keys are Hadamard-rotated at append time, so the query is rotated here too.
    static float score_page_keys(const void* page_bytes, const float* query_rows);
    // Scores the given pages against the accumulated query and returns them best-first.
    // `page_means[i]` points at the 32 KB summary of `absolute_pages[i]`; the summaries live in
    // per-block device allocations, so the caller passes one pointer per page instead of a
    // single contiguous array.
    std::vector<RetrievalCandidate> rank(const void* host_page_means,
                                         const std::vector<std::uint32_t>& absolute_pages,
                                         cudaStream_t stream);

    // Copies the ring summary of the pages [first_page, first_page + page_count) into
    // `destination` (same per-page layout, bf16). Pages never accumulated are zeroed.
    void copy_page_means(std::uint32_t first_page, std::uint32_t page_count, void* destination,
                         cudaStream_t stream);

    // Releases the ring slot of a page that has left the window without being archived.
    void forget(std::uint32_t absolute_page, cudaStream_t stream);

    // Debug helper: a rk4v4-e8 page holds packed offset-binary int4 keys, so a valid page must show
    // nibble values centred on 8 (code 0) rather than on 0. Prints a few decoded codes for one page.
    static void debug_scan_page_keys(const void* page_bytes, std::uint32_t absolute_page);

private:
    KVRetrieval();
    void ensure_allocated();

    float* means_          = nullptr;  // ring: [page][layer][kv_head][head_dim], fp32
    std::uint32_t* tags_   = nullptr;  // ring slot -> absolute page
    std::uint32_t* counts_ = nullptr;  // ring slot -> key rows folded in so far
    float* query_          = nullptr;  // [layer][kv_head][head_dim]
    float* frozen_query_   = nullptr;  // query_ snapshot taken when the prompt completes
    float* last_query_     = nullptr;  // query of the final prompt token only
    bool prompt_complete_  = false;
    std::uint32_t* query_rows_ = nullptr;
    float* scores_         = nullptr;  // [page] scratch for rank()
    float* scratch_        = nullptr;  // [pages_in_chunk] page indices resolved on the host

    // Host-side view of the current chunk.
    std::vector<std::uint32_t> chunk_pages_;
    std::uint32_t chunk_first_page_ = 0;
    std::int32_t chunk_tokens_      = 0;
    const std::int32_t* chunk_positions_host_ = nullptr;
    std::vector<std::int32_t> chunk_positions_;
    // Device copy of chunk_positions_ (the kernels read positions per token).
    std::int32_t* chunk_positions_device_ = nullptr;
    std::size_t chunk_positions_capacity_ = 0;
};

}  // namespace ninfer::models::qwen3_5::detail
