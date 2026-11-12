#pragma once

#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct KVArchiveStats {
    std::uint64_t blocks       = 0;
    std::uint64_t pages        = 0;
    std::uint64_t bytes        = 0;
    std::uint64_t evicted      = 0;
    std::uint64_t skipped      = 0;
    std::uint32_t oldest_token = 0;
    std::uint32_t newest_token = 0;
};

// Process-wide archive of KV pages that the resident window slid out (Leg B / kvmem W3-a).
//
// The pages are copied Device -> Host as raw bytes into a pinned arena; the absolute token range
// they cover is recorded so a later recall pass (W3-b) can put them back into free slots. The
// logical page state machine is untouched: archiving happens while the pages are still owned by
// the address space, immediately before `slide_prefix_pages` releases them.
//
// Disabled unless NINFER_KV_ARCHIVE=1. Capacity comes from NINFER_KV_ARCHIVE_MIB (default 2048).
class KVArchive {
public:
    struct Block {
        HostKVAllocation allocation;
        std::uint32_t begin_token = 0;
        std::uint32_t pages       = 0;
        // Host bf16 summary per page, [pages][kRetrievalPageMeanBytes / 2], empty when retrieval
        // indexing is off. Kept on the host so ranking can upload one contiguous block and cannot
        // race with a device buffer freed by eviction.
        std::vector<std::uint16_t> means_host;
    };

    KVArchive()  = default;
    ~KVArchive() = default;

    KVArchive(const KVArchive&)            = delete;
    KVArchive& operator=(const KVArchive&) = delete;
    KVArchive(KVArchive&&)                 = delete;
    KVArchive& operator=(KVArchive&&)      = delete;

    // nullptr when archiving is disabled by the environment.
    [[nodiscard]] static KVArchive* instance() noexcept;
    [[nodiscard]] static bool enabled() noexcept;

    // Copies `pages` (which must still be readable) into the pinned arena. `first_absolute_token` is
    // the absolute position of the first token covered by `pages[0]`. Async on `stream`.
    void archive(const DeviceKVPagePool& pool, std::span<const DeviceKVPageHandle> pages,
                 std::uint32_t first_absolute_token, cudaStream_t stream);

    [[nodiscard]] KVArchiveStats stats() const noexcept;

    [[nodiscard]] const std::vector<Block>& blocks() const noexcept { return blocks_; }

    // One page of archived KV, addressed by its true absolute token range.
    struct RecallPage {
        HostKVAllocationConstView source;
        std::uint32_t absolute_token = 0;
    };

    // Leg B / kvmem W3-b: resolve `page_count` page-aligned pages starting at `first_absolute_token`
    // to pinned sources that can be copied straight back into device slots. Pages whose range is not
    // (or no longer) archived are skipped, so the result can be shorter than requested. The caller
    // must keep the addresses stable: arena blocks are only freed by FIFO eviction.
    [[nodiscard]] std::vector<RecallPage> recall_pages(std::uint32_t first_absolute_token,
                                                       std::uint32_t page_count) const;

private:
    [[nodiscard]] bool ensure_arena(const HostKVPageLayout& layout) noexcept;

    std::unique_ptr<HostKVArena> arena_;
    std::optional<HostKVPageLayout> layout_;
    std::vector<Block> blocks_;
    KVArchiveStats stats_;
};

}  // namespace ninfer::models::qwen3_5::detail
