/******************************************************************************
 * HIP port of CUB's util_ptx.cuh (gfx1201, wave32).
 *
 * Every NVIDIA PTX asm body has been replaced with a HIP intrinsic or plain C++
 * equivalent. Function signatures are IDENTICAL to upstream CUB 1.15 so the rest
 * of CUB (BlockMergeSort, WarpMergeSort, BlockRadixRank, ...) compiles unchanged.
 *
 * Semantics notes for the shuffles:
 *   NVIDIA shfl.up/down/idx encode clamp bounds in `flags` (bits 0..7) plus the
 *   segment mask (bits 8..11). We decode the clamp from `flags & 0x1f` and
 *   implement the operation by computing an absolute source lane and using
 *   __shfl_sync, which is exactly equivalent for full and sub-warp segments.
 ******************************************************************************/
#pragma once

#include "../config.cuh"
#include "../util_type.cuh"
#include "../util_debug.cuh"
#include <cstdint>

CUB_NAMESPACE_BEGIN

// ------------------------------------------------------------------
// PTX register modifiers — unused on HIP, kept for source compatibility
// ------------------------------------------------------------------
#define _CUB_ASM_PTR_ "r"
#define _CUB_ASM_PTR_SIZE_ "u32"

/**
 * Shift-right then add.  Returns (\p x >> \p shift) + \p addend.
 */
__device__ __forceinline__ unsigned int SHR_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    return (x >> shift) + addend;
}

/**
 * Shift-left then add.  Returns (\p x << \p shift) + \p addend.
 */
__device__ __forceinline__ unsigned int SHL_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    return (x << shift) + addend;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Bitfield-extract (32-bit): bits [bit_start, bit_start+num_bits) of source.
 */
template <typename UnsignedBits, int BYTE_LEN>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<BYTE_LEN>      /*byte_len*/)
{
    unsigned int src = static_cast<unsigned int>(source);
    if (num_bits == 0u || bit_start >= 32u) { return 0u; }
    unsigned int mask = (num_bits >= 32u) ? 0xFFFFFFFFu : ((1u << num_bits) - 1u);
    return (src >> bit_start) & mask;
}

/**
 * Bitfield-extract for 64-bit types.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<8>             /*byte_len*/)
{
    const unsigned long long MASK = (num_bits >= 64u) ? ~0ull : ((1ull << num_bits) - 1);
    return static_cast<unsigned int>((source >> bit_start) & MASK);
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Bitfield-extract.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits source,
    unsigned int bit_start,
    unsigned int num_bits)
{
    return BFE(source, bit_start, num_bits, Int2Type<sizeof(UnsignedBits)>());
}

/**
 * \brief Bitfield insert: replace num_bits bits of \p x at \p bit_start with the
 *        num_bits least-significant bits of \p y (NVIDIA bfi.b32 semantics).
 */
__device__ __forceinline__ void BFI(
    unsigned int &ret,
    unsigned int x,
    unsigned int y,
    unsigned int bit_start,
    unsigned int num_bits)
{
    if (num_bits == 0u || bit_start >= 32u) { ret = x; return; }
    unsigned int mask = (num_bits >= 32u) ? 0xFFFFFFFFu : ((1u << num_bits) - 1u);
    const unsigned int shift = (bit_start + num_bits >= 32u) ? 0u : (32u - bit_start - num_bits);
    // mask over the replaced window, positioned at bit_start
    const unsigned int window = (mask << bit_start) & (0xFFFFFFFFu >> shift);
    ret = (x & ~window) | ((y & mask) << bit_start);
}

/**
 * \brief Three-operand add.  Returns \p x + \p y + \p z.
 */
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
    return x + y + z;
}

/**
 * \brief Byte-permute: pick four arbitrary bytes from two 32-bit registers.
 *        Bytes of {b, a} are numbered 0..7 = {b.b0,b.b1,b.b2,b.b3,a.b0,a.b1,a.b2,a.b3}?? —
 *        NVIDIA numbering: {b, a} = {{b7..b4}, {b3..b0}} means index nibble selects from
 *        the 8 bytes of the 64-bit value (b:a) where byte i = (i < 4) ? a.byte(i) : b.byte(i-4).
 *        Result byte j (j = 0..3, from low nibbles n0..n3 of index) = selected byte.
 *        Sign-extension modes (high bits of index) are not implemented (unused by CUB sort).
 */
__device__ __forceinline__ int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
    const unsigned char ab[8] = {
        static_cast<unsigned char>(a        & 0xFFu),
        static_cast<unsigned char>((a >> 8)  & 0xFFu),
        static_cast<unsigned char>((a >> 16) & 0xFFu),
        static_cast<unsigned char>((a >> 24) & 0xFFu),
        static_cast<unsigned char>(b        & 0xFFu),
        static_cast<unsigned char>((b >> 8)  & 0xFFu),
        static_cast<unsigned char>((b >> 16) & 0xFFu),
        static_cast<unsigned char>((b >> 24) & 0xFFu),
    };
    unsigned int ret = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned int nibble = (index >> (4 * j)) & 0xFu;
        // msb-mode (nibble bit 3 set) selects sign-replication of a byte's MSB — unused by CUB
        const unsigned int sel = nibble & 0x7u;
        unsigned char byte = (sel < 8u) ? ab[sel] : static_cast<unsigned char>(0u);
        ret |= static_cast<unsigned int>(byte) << (8 * j);
    }
    return static_cast<int>(ret);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Sync-threads barrier (named barrier with thread count). On HIP we fall back to a
 * full CTA barrier — correct whenever `count` equals the block size, which is how
 * CUB uses it (BlockRadixRank full-block barriers).
 */
__device__ __forceinline__ void BAR(int /*count*/)
{
    __syncthreads();
}

/**
 * CTA barrier
 */
__device__  __forceinline__ void CTA_SYNC()
{
    __syncthreads();
}

/**
 * CTA barrier with predicate
 */
__device__  __forceinline__ int CTA_SYNC_AND(int p)
{
    return __syncthreads_and(p);
}

/**
 * CTA barrier with predicate
 */
__device__  __forceinline__ int CTA_SYNC_OR(int p)
{
    return __syncthreads_or(p);
}

/**
 * Warp barrier — the shim's __syncwarp is a memory fence on AMD (no hardware
 * warp barrier); lanes in a wave stay convergent for CUB's usage patterns.
 */
__device__  __forceinline__ void WARP_SYNC(unsigned int /*member_mask*/)
{
    // no-op: matches upstream when CUB_USE_COOPERATIVE_GROUPS is undefined
}

/**
 * Warp any
 */
__device__  __forceinline__ int WARP_ANY(int predicate, unsigned int member_mask)
{
    return __any_sync(static_cast<unsigned long long>(member_mask), predicate);
}

/**
 * Warp all
 */
__device__  __forceinline__ int WARP_ALL(int predicate, unsigned int member_mask)
{
    return __all_sync(static_cast<unsigned long long>(member_mask), predicate);
}

/**
 * Warp ballot
 */
__device__  __forceinline__ int WARP_BALLOT(int predicate, unsigned int member_mask)
{
    return static_cast<int>(__ballot_sync(static_cast<unsigned long long>(member_mask), predicate != 0));
}

/**
 * Warp synchronous shfl_up.
 * NVIDIA semantics: lane i gets word from lane (i - src_offset); if that lane is
 * below `first_thread` (flags bits 0..7), the lane keeps its own value.
 */
__device__ __forceinline__
unsigned int SHFL_UP_SYNC(unsigned int word, int src_offset, int flags, unsigned int member_mask)
{
    const unsigned int lane = threadIdx.x & 31u;
    const int first_thread = flags & 0x1f;
    int src = static_cast<int>(lane) - src_offset;
    if (src < first_thread) { src = static_cast<int>(lane); }
    return __shfl_sync(static_cast<unsigned long long>(member_mask), word, src);
}

/**
 * Warp synchronous shfl_down.
 * NVIDIA semantics: lane i gets word from lane (i + src_offset); if that lane is
 * above `last_thread` (flags bits 0..7), the lane keeps its own value.
 */
__device__ __forceinline__
unsigned int SHFL_DOWN_SYNC(unsigned int word, int src_offset, int flags, unsigned int member_mask)
{
    const unsigned int lane = threadIdx.x & 31u;
    const int last_thread = flags & 0x1f;
    int src = static_cast<int>(lane) + src_offset;
    if (src > last_thread) { src = static_cast<int>(lane); }
    return __shfl_sync(static_cast<unsigned long long>(member_mask), word, src);
}

/**
 * Warp synchronous shfl_idx (broadcast within the logical warp segment).
 * NVIDIA semantics: lane i gets word from the segment base + src_lane; out-of-range
 * src_lane keeps the lane's own value. flags bits 0..7 hold (LOGICAL_WARP_THREADS-1).
 */
__device__ __forceinline__
unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane, int flags, unsigned int member_mask)
{
    const unsigned int lane = threadIdx.x & 31u;
    const int last = flags & 0x1f;                       // LOGICAL_WARP_THREADS - 1
    const int base = static_cast<int>(lane) & ~last;     // segment base
    if (src_lane < 0 || src_lane > last) { return word; }
    return __shfl_sync(static_cast<unsigned long long>(member_mask), word, base + src_lane);
}

/**
 * Warp synchronous shfl_idx (member_mask only, full warp).
 */
__device__ __forceinline__
unsigned int SHFL_IDX_SYNC(unsigned int word, int src_lane, unsigned int member_mask)
{
    return __shfl_sync(static_cast<unsigned long long>(member_mask), word, src_lane);
}

/**
 * Floating point multiply. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FMUL_RZ(float a, float b)
{
    return __fmul_rn(a, b); // HIP has no _rz; round-to-nearest is fine for CUB's uses
}

/**
 * Floating point multiply-add. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
{
    return __fmaf_rn(a, b, c); // HIP has no _rz
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Terminates the calling thread (unused on the MergeSort paths; safe no-op).
 */
__device__ __forceinline__ void ThreadExit() {
    // AMD: no per-thread exit primitive reachable from CUB's usage; kept as no-op.
}

/**
 * \brief  Abort execution and generate an interrupt to the host CPU
 */
__device__ __forceinline__ void ThreadTrap() {
    __builtin_trap();
}

/**
 * \brief Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId()
{
    return threadIdx.x & 31u;
}

/**
 * \brief Returns the row-major linear thread identifier for a multidimensional thread block
 */
__device__ __forceinline__ int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y)) +
            ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) +
            threadIdx.x;
}

/**
 * \brief Returns the warp ID of the calling thread (CTA-linear, 1D blocks).
 */
__device__ __forceinline__ unsigned int WarpId()
{
    return threadIdx.x >> 5u;
}

/**
 * @brief Returns the warp mask for a warp of @p LOGICAL_WARP_THREADS threads
 */
template <int LOGICAL_WARP_THREADS,
          int PTX_ARCH = CUB_PTX_ARCH>
__host__ __device__ __forceinline__
unsigned int WarpMask(unsigned int warp_id)
{
  constexpr bool is_pow_of_two = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;
  constexpr bool is_arch_warp = LOGICAL_WARP_THREADS ==
                                CUB_WARP_THREADS(PTX_ARCH);

  unsigned int member_mask =
    0xFFFFFFFFu >> (CUB_WARP_THREADS(PTX_ARCH) - LOGICAL_WARP_THREADS);

  if (is_pow_of_two && !is_arch_warp)
  {
    member_mask <<= warp_id * LOGICAL_WARP_THREADS;
  }

  return member_mask;
}

/**
 * \brief Returns the warp lane mask of all lanes less than the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskLt()
{
    const unsigned int lane = threadIdx.x & 31u;
    return (1u << lane) - 1u;
}

/**
 * \brief Returns the warp lane mask of all lanes less than or equal to the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskLe()
{
    const unsigned int lane = threadIdx.x & 31u;
    return (lane >= 31u) ? 0xFFFFFFFFu : ((2u << lane) - 1u);
}

/**
 * \brief Returns the warp lane mask of all lanes greater than the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskGt()
{
    const unsigned int lane = threadIdx.x & 31u;
    return (lane >= 31u) ? 0u : (0xFFFFFFFFu << (lane + 1u));
}

/**
 * \brief Returns the warp lane mask of all lanes greater than or equal to the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskGe()
{
    const unsigned int lane = threadIdx.x & 31u;
    return 0xFFFFFFFFu << lane;
}

/** @} */       // end group UtilPtx




/**
 * \brief Shuffle-up for any data type.
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleUp(
    T               input,
    int             src_offset,
    int             first_thread,
    unsigned int    member_mask)
{
    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    #pragma unroll
    for (int WORD = 0; WORD < WORDS; ++WORD)
    {
        output_alias[WORD] = SHFL_UP_SYNC(static_cast<unsigned int>(input_alias[WORD]),
                                          src_offset, first_thread, member_mask);
    }

    return output;
}


/**
 * \brief Shuffle-down for any data type.
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleDown(
    T               input,
    int             src_offset,
    int             last_thread,
    unsigned int    member_mask)
{
    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    #pragma unroll
    for (int WORD = 0; WORD < WORDS; ++WORD)
    {
        output_alias[WORD] = SHFL_DOWN_SYNC(static_cast<unsigned int>(input_alias[WORD]),
                                            src_offset, last_thread, member_mask);
    }

    return output;
}


/**
 * \brief Shuffle-broadcast for any data type.
 */
template <
    int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
    typename T>
__device__ __forceinline__ T ShuffleIndex(
    T               input,
    int             src_lane,
    unsigned int    member_mask)
{
    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    // flags = ((32 - LOGICAL_WARP_THREADS) << 8) | (LOGICAL_WARP_THREADS - 1)
    const int flags = ((32 - LOGICAL_WARP_THREADS) << 8) | (LOGICAL_WARP_THREADS - 1);

    #pragma unroll
    for (int WORD = 0; WORD < WORDS; ++WORD)
    {
        output_alias[WORD] = SHFL_IDX_SYNC(static_cast<unsigned int>(input_alias[WORD]),
                                           src_lane, flags, member_mask);
    }

    return output;
}



/**
 * Compute a 32b mask of threads having the same least-significant
 * LABEL_BITS of \p label as the calling thread.
 */
template <int LABEL_BITS>
inline __device__ unsigned int MatchAny(unsigned int label)
{
    unsigned int retval = 0u;

    // Extract masks of common threads for each bit
    #pragma unroll
    for (int BIT = 0; BIT < LABEL_BITS; ++BIT)
    {
        const unsigned int current_bit = 1u << BIT;
        const bool has_bit = (label & current_bit) == current_bit;
        unsigned int mask = static_cast<unsigned int>(
            __ballot_sync(0xFFFFFFFFull, has_bit));
        // Remove peers who differ (invert when this lane's bit is clear)
        if (!has_bit) { mask = ~mask; }
        retval = (BIT == 0) ? mask : (retval & mask);
    }

    return retval;
}

CUB_NAMESPACE_END
