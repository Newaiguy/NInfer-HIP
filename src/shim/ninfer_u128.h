#pragma once
// NInfer -> HIP port shim: 128-bit unsigned integer for host code.
// Clang/hipcc: native unsigned __int128. MSVC x64 (no __int128): hi/lo struct with
// _umul128 and explicit operators covering the operations ninfer uses
// (mul by uint64, +uint64, <<, >>, compares, cast to uint64).
#include <cstdint>

#if defined(__SIZEOF_INT128__)
using ninfer_u128 = unsigned __int128;
#else
#include <intrin.h>

struct ninfer_u128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    constexpr ninfer_u128() = default;
    constexpr ninfer_u128(std::uint64_t value) : lo(value), hi(0) {}
    constexpr ninfer_u128(std::uint64_t l, std::uint64_t h) : lo(l), hi(h) {}

    static ninfer_u128 mul(std::uint64_t a, std::uint64_t b) {
        ninfer_u128 r;
        r.lo = _umul128(a, b, &r.hi);
        return r;
    }

    friend ninfer_u128 operator*(const ninfer_u128& l, std::uint64_t r) {
        const ninfer_u128 low_part  = mul(l.lo, r);
        const ninfer_u128 high_part = mul(l.hi, r);
        return ninfer_u128{low_part.lo, low_part.hi + high_part.lo};
    }

    friend ninfer_u128 operator+(const ninfer_u128& l, std::uint64_t r) {
        ninfer_u128 out{l.lo + r, l.hi};
        if (out.lo < l.lo) { ++out.hi; }
        return out;
    }

    constexpr ninfer_u128 operator<<(unsigned s) const {
        if (s == 0) { return *this; }
        if (s < 64) { return ninfer_u128{lo << s, (hi << s) | (lo >> (64 - s))}; }
        return ninfer_u128{0, lo << (s - 64)};
    }

    constexpr ninfer_u128 operator>>(unsigned s) const {
        if (s == 0) { return *this; }
        if (s < 64) { return ninfer_u128{(lo >> s) | (hi << (64 - s)), hi >> s}; }
        return ninfer_u128{hi >> (s - 64), 0};
    }

    friend constexpr bool operator==(const ninfer_u128& l, const ninfer_u128& r) {
        return l.lo == r.lo && l.hi == r.hi;
    }
    friend constexpr bool operator!=(const ninfer_u128& l, const ninfer_u128& r) { return !(l == r); }
    friend constexpr bool operator<(const ninfer_u128& l, const ninfer_u128& r) {
        return l.hi != r.hi ? l.hi < r.hi : l.lo < r.lo;
    }
    friend constexpr bool operator>(const ninfer_u128& l, const ninfer_u128& r) { return r < l; }
    friend constexpr bool operator<=(const ninfer_u128& l, const ninfer_u128& r) { return !(l > r); }
    friend constexpr bool operator>=(const ninfer_u128& l, const ninfer_u128& r) { return !(l < r); }
    friend constexpr bool operator==(const ninfer_u128& l, std::uint64_t r) { return l == ninfer_u128(r); }
    friend constexpr bool operator>(const ninfer_u128& l, std::uint64_t r) { return l > ninfer_u128(r); }

    constexpr operator std::uint64_t() const { return lo; }
};
#endif
