#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "ninfer/ops/scatter.h"
#include "ops/linear/ternary/ternary_dispatch.h" // AMD port: verify-mode kernel selection
#include <windows.h>
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_5::execution {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    // AMD port: this forward pass is the speculative verify. Bracketing it lets the ternary
    // dispatch pick the faster small-T kernel for it without changing prefill numerics
    // (the two are the same shape, so the shape alone cannot distinguish them).
    ops::detail::ScopedTernaryVerifyMode verify_mode;
    LARGE_INTEGER tv0 = {}, tv1 = {}, tv2 = {}, tv3 = {}, tv4 = {};
    QueryPerformanceCounter(&tv0);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    QueryPerformanceCounter(&tv1);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    // TEMP: dump drafts vs target tokens (MTP verify all-reject diagnosis).
    if (std::getenv("NINFER_MTP_DUMP") != nullptr && std::getenv("NINFER_MTP_DUMP")[0] == '1') {
        const std::int32_t nd = frame.drafts.ne[0] > 8 ? 8 : frame.drafts.ne[0];
        std::vector<std::int32_t> hd(static_cast<std::size_t>(nd));
        std::vector<std::int32_t> ht(4, -1);
        cudaMemcpyAsync(hd.data(), frame.drafts.data, static_cast<std::size_t>(nd) * 4,
                        cudaMemcpyDeviceToHost, execution.device.stream);
        cudaMemcpyAsync(ht.data(), frame.target_tokens.data, 4 * 4, cudaMemcpyDeviceToHost,
                        execution.device.stream);
        cudaStreamSynchronize(execution.device.stream);
        std::fprintf(stderr, "[mtp-dump] drafts=%d,%d,%d,%d,%d,%d,%d,%d target=%d,%d,%d,%d\n",
                     nd > 0 ? hd[0] : -1, nd > 1 ? hd[1] : -1, nd > 2 ? hd[2] : -1,
                     nd > 3 ? hd[3] : -1, nd > 4 ? hd[4] : -1, nd > 5 ? hd[5] : -1,
                     nd > 6 ? hd[6] : -1, nd > 7 ? hd[7] : -1, ht[0], ht[1], ht[2], ht[3]);
        std::fflush(stderr);
    }
    if (frame.proposal_q.data != nullptr) {
        // AMD port: NINFER_ACCEPT_SINGLE=1 forces the warp-greedy sparse accept (no
        // multiblock partial/group pipeline, which is broken on gfx1201). The sparse
        // stochastic route is unavailable under this switch; the greedy warp accept is
        // exact for temperature<=0 without penalties and is the shape used by DFlash2.
        static const bool force_single = std::getenv("NINFER_ACCEPT_MULTIBLOCK") == nullptr;
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            {force_single}, execution.work, execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            execution.work, execution.device.stream);
    }
    QueryPerformanceCounter(&tv2);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    QueryPerformanceCounter(&tv3);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
    QueryPerformanceCounter(&tv4);
    {
        static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
        static double a_v = 0, a_a = 0, a_s = 0, a_sc = 0, a_t = 0;
        static int cnt = 0;
        a_v += static_cast<double>(tv1.QuadPart - tv0.QuadPart) * 1000.0 / freq.QuadPart;
        a_a += static_cast<double>(tv2.QuadPart - tv1.QuadPart) * 1000.0 / freq.QuadPart;
        a_s += static_cast<double>(tv3.QuadPart - tv2.QuadPart) * 1000.0 / freq.QuadPart;
        a_sc += static_cast<double>(tv4.QuadPart - tv3.QuadPart) * 1000.0 / freq.QuadPart;
        a_t += static_cast<double>(tv4.QuadPart - tv0.QuadPart) * 1000.0 / freq.QuadPart;
        if (++cnt % 16 == 0) {
            std::fprintf(stderr, "[verify-timing] %d calls: run_layers=%.1f accept=%.1f select=%.1f scatter=%.1f total=%.1f ms (avg %.1f/call)\n",
                         cnt, a_v, a_a, a_s, a_sc, a_t, a_t / 16);
            std::fflush(stderr);
            a_v = a_a = a_s = a_sc = a_t = 0;
        }
    }
}

} // namespace ninfer::models::qwen3_5::execution
