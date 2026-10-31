#pragma once
// NInfer -> HIP port shim: POSIX fd I/O over Win32 for MSVC host builds.
// Covers what ninfer's artifact file_io + context_cost actually use:
//   open(O_RDONLY|O_CLOEXEC[|O_DIRECT]), fstat, pread, close, getpid.
// fds are indices into a Win32 HANDLE table (HANDLE is 64-bit, fd int).
// O_DIRECT maps to FILE_FLAG_NO_BUFFERING; callers guarantee sector alignment.
#include <windows.h>

#include <cstdint>
#include <mutex>
#include <vector>

#ifndef O_DIRECT
#define O_DIRECT 0x00400000
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

#if !defined(_SSIZE_T_DEFINED)
typedef long long ssize_t;
#define _SSIZE_T_DEFINED
#endif

#ifndef S_ISREG
#define S_ISREG(m) (((m) & 0xF000) == 0x8000)
#endif

// Minimal stat replacement (UCRT off_t/struct stat st_size is 32-bit here, which would
// truncate the >2GiB model artifacts; this carries a full 64-bit size).
struct shim_stat {
    long long st_size;
    unsigned long st_mode;
};
// MSVC sys/stat.h S_ISREG(m) tests (m & _S_IFMT) == _S_IFREG with _S_IFREG = 0x8000.

namespace ninfer_shim_posix {

struct Entry {
    HANDLE handle;
};

inline std::vector<Entry>& table() {
    static std::vector<Entry> t;
    return t;
}

inline std::mutex& mutex() {
    static std::mutex m;
    return m;
}

} // namespace ninfer_shim_posix

inline int open(const char* path, int flags) {
    DWORD attrs = FILE_ATTRIBUTE_NORMAL;
    if ((flags & O_DIRECT) != 0) { attrs |= FILE_FLAG_NO_BUFFERING; }
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, attrs,
                           nullptr);
    if (h == INVALID_HANDLE_VALUE) { return -1; }
    std::lock_guard<std::mutex> lock(ninfer_shim_posix::mutex());
    auto& t = ninfer_shim_posix::table();
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i].handle == nullptr) {
            t[i].handle = h;
            return static_cast<int>(i);
        }
    }
    t.push_back(ninfer_shim_posix::Entry{h});
    return static_cast<int>(t.size() - 1);
}

// std::filesystem::path::value_type is wchar_t on Windows
inline int open(const wchar_t* path, int flags) {
    DWORD attrs = FILE_ATTRIBUTE_NORMAL;
    if ((flags & O_DIRECT) != 0) { attrs |= FILE_FLAG_NO_BUFFERING; }
    HANDLE h = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, attrs,
                           nullptr);
    if (h == INVALID_HANDLE_VALUE) { return -1; }
    std::lock_guard<std::mutex> lock(ninfer_shim_posix::mutex());
    auto& t = ninfer_shim_posix::table();
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i].handle == nullptr) {
            t[i].handle = h;
            return static_cast<int>(i);
        }
    }
    t.push_back(ninfer_shim_posix::Entry{h});
    return static_cast<int>(t.size() - 1);
}

inline int close(int fd) {
    std::lock_guard<std::mutex> lock(ninfer_shim_posix::mutex());
    auto& t = ninfer_shim_posix::table();
    if (fd < 0 || static_cast<std::size_t>(fd) >= t.size() || t[static_cast<std::size_t>(fd)].handle == nullptr) {
        return -1;
    }
    CloseHandle(t[static_cast<std::size_t>(fd)].handle);
    t[static_cast<std::size_t>(fd)].handle = nullptr;
    return 0;
}

// Positional read: OVERLAPPED ReadFile never moves the file pointer (pread semantics).
inline long long pread(int fd, void* buffer, unsigned long long count, long long offset) {
    HANDLE h = ninfer_shim_posix::table()[static_cast<std::size_t>(fd)].handle;
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xffffffffull);
    overlapped.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xffffffffull);
    DWORD got             = 0;
    if (count > 0xffffffffull) { errno = EINVAL; return -1; }
    if (!ReadFile(h, buffer, static_cast<DWORD>(count), &got, &overlapped)) {
        const DWORD error = GetLastError();
        if (error == ERROR_HANDLE_EOF || error == ERROR_BROKEN_PIPE) { return static_cast<long long>(got); }
        switch (error) {
        case ERROR_NOACCESS:              errno = EFAULT; break;
        case ERROR_INVALID_PARAMETER:     errno = EINVAL; break;
        case ERROR_INVALID_USER_BUFFER:
        case ERROR_WORKING_SET_QUOTA:
        case ERROR_NOT_ENOUGH_MEMORY:     errno = ENOMEM; break;
        default:                          errno = EIO; break;
        }
        fprintf(stderr, "[shim] pread fail fd=%d count=%llu offset=%lld gle=%lu\n",
                fd, count, offset, error);
        return -1;
    }
    return static_cast<long long>(got);
}

inline int fstat(int fd, shim_stat* out) {
    HANDLE h = ninfer_shim_posix::table()[static_cast<std::size_t>(fd)].handle;
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(h, &size)) { return -1; }
    out->st_size = size.QuadPart;
    out->st_mode = 0x8000ul; // _S_IFREG
    return 0;
}

inline int getpid() { return static_cast<int>(GetCurrentProcessId()); }
