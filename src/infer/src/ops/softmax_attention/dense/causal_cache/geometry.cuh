#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    // gfx1201 long-context port: the stock 85-split ceiling is sized for a 170-SM part. On this
    // 32-CU card it is the cap that decides how many 64-key tiles one CTA has to walk, and at
    // 128k tokens that is 24 tiles per CTA against 9 at 48k: measured decode attention went from
    // 6.2ms to 66ms while the key count only grew 2.6x, because a long per-CTA ring loses the
    // prefetch window. 256 is the merge kernel's thread-count ceiling (static_assert in
    // small_t.cuh), so the T=1 geometry is raised to it and the ring stays short at long context.
    static constexpr int SmallTMaximumSplits = SmallTSplitScale == 1 ? 256 : 85 * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;

} // namespace ninfer::ops
