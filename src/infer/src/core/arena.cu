#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
    if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
    const std::size_t mask         = align - 1;
    const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
    return with_mask & ~static_cast<std::uintptr_t>(mask);
}

bool alloc_log_enabled() {
    static const bool on = [] { const char* e = std::getenv("NINFER_ALLOC_LOG"); return e != nullptr && e[0] == '1'; }();
    return on;
}

void log_alloc(const void* ptr, std::size_t bytes) {
    if (!alloc_log_enabled()) { return; }
    const auto a = reinterpret_cast<std::uintptr_t>(ptr);
    std::fprintf(stdout,
                 "[alloc] ptr=%p bytes=%zu mod2M=%llu mod16M=%llu mod256M=%llu mod1G=%llu mod4G=%llu\n",
                 ptr, bytes,
                 static_cast<unsigned long long>(a % (2u << 20)),
                 static_cast<unsigned long long>(a % (16u << 20)),
                 static_cast<unsigned long long>(a % (256u << 20)),
                 static_cast<unsigned long long>(a % (1ull << 30)),
                 static_cast<unsigned long long>(a % (4ull << 30)));
    std::fflush(stdout);
}
// gfx1201 long-context port: per-arena read bandwidth probe. The 126k decode step time has
// been observed to swing 43ms <-> 100ms between identical runs while the standalone d2d probe
// stays healthy, which points at the physical placement quality of the big arenas rather than at
// the kernels. This sweep reads the freshly allocated region once and prints the achieved read
// bandwidth, so every run carries a fingerprint of the memory it actually got.
__global__ void arena_read_sweep_kernel(const uint4* __restrict__ p, unsigned long long n4,
                                       unsigned int* __restrict__ sink) {
    unsigned long long i          = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long step = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    unsigned int acc              = 0u;
    for (; i < n4; i += step) {
        const uint4 v = p[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) { *sink = acc; }
}

bool alloc_bw_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("NINFER_ALLOC_BW");
        return e != nullptr && e[0] != '0';
    }();
    return on;
}

void probe_arena_bw(const char* tag, void* ptr, std::size_t bytes) {
    if (!alloc_bw_enabled() || ptr == nullptr || bytes < (1u << 20)) { return; }
    static unsigned int* sink = nullptr;
    if (sink == nullptr) {
        if (cudaMalloc(reinterpret_cast<void**>(&sink), sizeof(unsigned int)) != cudaSuccess) { return; }
    }
    const std::size_t n4 = bytes / sizeof(uint4);
    cudaEvent_t t0 = nullptr;
    cudaEvent_t t1 = nullptr;
    if (cudaEventCreate(&t0) != cudaSuccess || cudaEventCreate(&t1) != cudaSuccess) { return; }
    // Sweep the region three times: the first pass pays the WDDM residency/commit cost, the
    // later passes measure steady-state read throughput of whatever the driver actually bound
    // the allocation to (VRAM vs shared system memory). Both numbers matter.
    double gbs[3] = {0.0, 0.0, 0.0};
    cudaError_t err = cudaSuccess;
    for (int pass = 0; pass < 3; ++pass) {
        cudaEventRecord(t0);
        arena_read_sweep_kernel<<<1024, 256>>>(reinterpret_cast<const uint4*>(ptr),
                                               static_cast<unsigned long long>(n4), sink);
        cudaEventRecord(t1);
        err          = cudaEventSynchronize(t1);
        float ms     = 0.0f;
        if (err == cudaSuccess) { cudaEventElapsedTime(&ms, t0, t1); }
        if (err == cudaSuccess && ms > 0.0f) {
            gbs[pass] = (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
        }
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    if (err != cudaSuccess) {
        std::fprintf(stdout, "[alloc-bw] %s bytes=%zu probe_failed\n", tag, bytes);
        std::fflush(stdout);
        return;
    }
    std::fprintf(stdout, "[alloc-bw] %s ptr=%p bytes=%zu read_GBs=%.1f/%.1f/%.1f\n", tag, ptr, bytes,
                 gbs[0], gbs[1], gbs[2]);
    std::fflush(stdout);
}

void free_device(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFree", cudaFree(ptr));
        ptr = nullptr;
    }
}

void free_pinned(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFreeHost", cudaFreeHost(ptr));
        ptr = nullptr;
    }
}

} // namespace

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }
    p = ptr;
    log_alloc(p, bytes);
    probe_arena_bw("buffer", p, bytes);
}

DeviceBuffer::~DeviceBuffer() { free_device(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
    other.p     = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_device(p);
    p     = other.p;
    bytes = other.bytes;

    other.p     = nullptr;
    other.bytes = 0;
    return *this;
}

void DeviceBuffer::fill(int byte_value) {
    if (bytes == 0) { return; }
    const cudaError_t err = cudaMemset(p, byte_value, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
    }
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
    require_range(byte_offset, count, "host-to-device copy");
    if (count == 0) { return; }
    void* destination     = static_cast<std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy host-to-device failed", err));
    }
    // Pageable H2D copies may return before the device transfer completes. Settle the
    // default stream before callers submit consumers on non-blocking streams.
    const cudaError_t settled = cudaStreamSynchronize(nullptr);
    if (settled != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("host-to-device copy synchronization failed", settled));
    }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count,
                                std::size_t byte_offset) const {
    require_range(byte_offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const void* source    = static_cast<const std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy device-to-host failed", err));
    }
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count,
                                 const char* operation) const {
    if (byte_offset > bytes || count > bytes - byte_offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.off_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ != nullptr && saved_offset_ <= arena_->off_) { arena_->off_ = saved_offset_; }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }

    log_alloc(ptr, capacity_bytes);
    probe_arena_bw("arena", ptr, capacity_bytes);
    base_ = ptr;
    cap_  = capacity_bytes;
    off_  = 0;
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), cap_(storage.bytes), owns_(false) {
    if (base_ == nullptr || cap_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() {
    if (owns_) { free_device(base_); }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_) {
    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    if (owns_) { free_device(base_); }
    base_ = other.base_;
    cap_  = other.cap_;
    off_  = other.off_;
    peak_ = other.peak_;
    owns_ = other.owns_;

    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
    return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("DeviceArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }

    const std::uintptr_t base_addr           = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t current_addr        = checked_add_uintptr(base_addr, off_);
    const std::uintptr_t aligned_addr        = align_up_addr(current_addr, align);
    const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
    if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("arena aligned offset overflows size_t");
    }
    const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
        throw std::overflow_error("arena allocation end offset overflows size_t");
    }
    const std::size_t end = aligned_offset + bytes;
    if (end > cap_) { throw std::bad_alloc(); }

    auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
    off_      = end;
    if (off_ > peak_) { peak_ = off_; }
    return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }

void DeviceArena::reset() noexcept { off_ = 0; }

void* DeviceArena::base() const noexcept { return base_; }

std::size_t DeviceArena::used() const noexcept { return off_; }

std::size_t DeviceArena::capacity() const noexcept { return cap_; }

std::size_t DeviceArena::peak_used() const noexcept { return peak_; }

void DeviceArena::reset_peak() noexcept { peak_ = off_; }

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMallocHost failed", err));
    }

    data_ = ptr;
    size_ = size_bytes;
}

PinnedHostBuffer::~PinnedHostBuffer() { free_pinned(data_); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_pinned(data_);
    data_ = other.data_;
    size_ = other.size_;

    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void* PinnedHostBuffer::data() const noexcept { return data_; }

std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

} // namespace ninfer
