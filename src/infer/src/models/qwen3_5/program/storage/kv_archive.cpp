#include "models/qwen3_5/program/storage/kv_archive.h"
#include "models/qwen3_5/program/storage/kv_retrieval.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

namespace ninfer::models::qwen3_5::detail {
namespace {

[[nodiscard]] std::size_t archive_budget_bytes() {
    static const std::size_t bytes = [] {
        const char* mib = std::getenv("NINFER_KV_ARCHIVE_MIB");
        const unsigned long value =
            (mib != nullptr && mib[0] != '\0') ? std::strtoul(mib, nullptr, 10) : 2048UL;
        return static_cast<std::size_t>(value) << 20;
    }();
    return bytes;
}

[[nodiscard]] bool archive_requested() {
    static const bool on = [] {
        const char* env = std::getenv("NINFER_KV_ARCHIVE");
        return env != nullptr && env[0] != '\0' && std::strtoul(env, nullptr, 10) != 0;
    }();
    return on;
}

[[nodiscard]] bool trace_enabled() {
    static const bool on = [] {
        const char* env = std::getenv("NINFER_KV_ARCHIVE_TRACE");
        return env != nullptr && env[0] != '\0' && std::strtoul(env, nullptr, 10) != 0;
    }();
    return on;
}

}  // namespace

bool KVArchive::enabled() noexcept { return archive_requested(); }

KVArchive* KVArchive::instance() noexcept {
    if (!archive_requested()) { return nullptr; }
    static KVArchive archive;
    return &archive;
}

bool KVArchive::ensure_arena(const HostKVPageLayout& layout) noexcept {
    if (arena_ != nullptr) { return layout_ == layout; }
    try {
        const HostKVPageLayout layouts[] = {layout};
        arena_ = std::make_unique<HostKVArena>(archive_budget_bytes(), layouts);
        layout_ = layout;
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[kv-archive] arena up: %zu MiB, page_stride=%zu, page_tokens=%d\n",
                         archive_budget_bytes() >> 20, layout.page_stride,
                         static_cast<int>(kPagedKVPageSize));
        }
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[kv-archive] arena init failed: %s\n", error.what());
        arena_.reset();
        layout_.reset();
        return false;
    }
}

void KVArchive::archive(const DeviceKVPagePool& pool, std::span<const DeviceKVPageHandle> pages,
                        std::uint32_t first_absolute_token, cudaStream_t stream) {
    if (pages.empty()) { return; }

    const HostKVPageLayout layout = plan_host_kv_page_layout(pool.geometry());
    if (!ensure_arena(layout)) {
        stats_.skipped += pages.size();
        return;
    }
    if (arena_ == nullptr) {
        stats_.skipped += pages.size();
        return;
    }

    const std::uint32_t page_count = static_cast<std::uint32_t>(pages.size());

    // Make room by dropping the oldest archived blocks (FIFO). A D2H may still be in flight on
    // `stream`, so synchronize before the arena hands those bytes back to the allocator.
    while (!arena_->can_allocate(layout, page_count) && !blocks_.empty()) {
        if (stream != nullptr) { (void)cudaStreamSynchronize(stream); }
        if (blocks_.front().means != nullptr) { (void)cudaFree(blocks_.front().means); }
        blocks_.erase(blocks_.begin());
        ++stats_.evicted;
    }

    std::optional<HostKVAllocation> allocation = arena_->allocate(layout, page_count);
    if (!allocation.has_value() || !allocation->valid()) {
        stats_.skipped += page_count;
        if (trace_enabled()) {
            std::fprintf(stderr, "[kv-archive] skip %u pages at token %u (arena free %zu B)\n",
                         page_count, first_absolute_token, arena_->free_bytes());
        }
        return;
    }

    pool.copy_to_host(pages, arena_->writable_view(*allocation), stream);

    Block block;
    block.pages       = page_count;
    block.begin_token = first_absolute_token;
    const std::uint64_t bytes = static_cast<std::uint64_t>(page_count) * layout.page_stride;
    const std::uint32_t end_token =
        first_absolute_token + page_count * static_cast<std::uint32_t>(kPagedKVPageSize);
    block.allocation = std::move(*allocation);

    // Keep a device copy of the mean key of every page so a later recall pass (W4) can rank
    // archived blocks against the current query without re-reading the window.
    if (KVRetrieval::instance() != nullptr) {
        void* means = nullptr;
        const std::size_t mean_bytes =
            static_cast<std::size_t>(page_count) * kRetrievalPageMeanBytes;
        if (cudaMalloc(&means, mean_bytes) == cudaSuccess) {
            KVRetrieval::instance()->copy_page_means(
                first_absolute_token / static_cast<std::uint32_t>(kPagedKVPageSize), page_count,
                means, stream);
            block.means = means;
        } else {
            (void)cudaGetLastError();
        }
    }

    if (blocks_.empty()) {
        stats_.oldest_token = block.begin_token;
    } else {
        stats_.oldest_token = std::min(stats_.oldest_token, block.begin_token);
    }
    stats_.newest_token = std::max(stats_.newest_token, end_token);
    ++stats_.blocks;
    stats_.pages += page_count;
    stats_.bytes += bytes;
    blocks_.push_back(std::move(block));

    if (trace_enabled()) {
        std::fprintf(stderr,
                     "[kv-archive] block %llu: %u pages, tokens [%u,%u), %llu B, arena free %zu B\n",
                     static_cast<unsigned long long>(stats_.blocks - 1), page_count,
                     first_absolute_token, end_token, static_cast<unsigned long long>(bytes),
                     arena_->free_bytes());
    }
}

KVArchiveStats KVArchive::stats() const noexcept { return stats_; }

std::vector<KVArchive::RecallPage> KVArchive::recall_pages(std::uint32_t first_absolute_token,
                                                           std::uint32_t page_count) const {
    std::vector<RecallPage> pages;
    if (arena_ == nullptr || !layout_.has_value() || page_count == 0 || blocks_.empty()) {
        return pages;
    }
    const std::uint32_t page_size = static_cast<std::uint32_t>(kPagedKVPageSize);
    pages.reserve(page_count);
    for (std::uint32_t index = 0; index < page_count; ++index) {
        const std::uint32_t token = first_absolute_token + index * page_size;
        const Block* owner        = nullptr;
        std::uint32_t offset       = 0;
        // Blocks are appended in slide order and never reshuffled, so a linear scan from the newest
        // end finds the freshest copy of a range that was archived twice.
        for (auto block = blocks_.rbegin(); block != blocks_.rend(); ++block) {
            const std::uint32_t span = block->pages * page_size;
            if (token >= block->begin_token && token < block->begin_token + span) {
                owner  = &*block;
                offset = (token - block->begin_token) / page_size;
                break;
            }
        }
        if (owner == nullptr) { break; }
        RecallPage page;
        page.source         = arena_->view(owner->allocation).subview(offset, 1);
        page.absolute_token = token;
        pages.push_back(page);
    }
    return pages;
}

}  // namespace ninfer::models::qwen3_5::detail
