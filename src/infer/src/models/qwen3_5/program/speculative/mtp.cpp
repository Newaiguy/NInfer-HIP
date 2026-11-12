#include "ninfer/ops/speculative_round.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "core/nvtx.h"
#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <windows.h> // AMD port: QPC MTP timer

namespace ninfer::models::qwen3_5::execution {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpCausalAttentionEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }
        // AMD port: MTP round section timing (NINFER_PROFILE_DECODE=1)
        struct MtpTimer {
            bool on = false;
            LARGE_INTEGER freq = {};
            LARGE_INTEGER t0 = {};
            double verify = 0.0, draft_ar = 0.0, prepare = 0.0, total = 0.0;
            double batch_fwd = 0.0, propose = 0.0, ar_fwd = 0.0, ar_prop = 0.0, ar_copy = 0.0;
            int rounds = 0;
            long accepted = 0;
        };
        static MtpTimer mt = [] {
            MtpTimer d;
            const char* e = std::getenv("NINFER_PROFILE_DECODE");
            d.on = e != nullptr && e[0] == '1';
            QueryPerformanceFrequency(&d.freq);
            return d;
        }();
        LARGE_INTEGER mt_start = {};
        if (mt.on) { QueryPerformanceCounter(&mt_start); mt.t0 = mt_start; }
        auto mt_mark = [&](double& acc) {
            if (!mt.on) { return; }
            CUDA_CHECK(cudaDeviceSynchronize());
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            acc += static_cast<double>(t.QuadPart - mt.t0.QuadPart) * 1000.0 / static_cast<double>(mt.freq.QuadPart);
            mt.t0 = t;
        };
        // Split marker used inside the draft section, where the interesting question is
        // how much of the per-step cost is the MTP layer forward versus the proposal head.
        auto mt_split = [&](double& acc) {
            if (!mt.on) { return; }
            CUDA_CHECK(cudaDeviceSynchronize());
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            acc += static_cast<double>(t.QuadPart - mt.t0.QuadPart) * 1000.0 / static_cast<double>(mt.freq.QuadPart);
            mt.t0 = t;
        };

        qwen3_5::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        // AMD port diagnosis: host_ingress is plain host memory, so dumping it here is safe
        // inside graph capture (device reads during capture are stale). Round counter limits noise.
        {
            static const bool host_dbg = [] {
                const char* e = std::getenv("NINFER_MTP_HOSTDBG");
                return e != nullptr && e[0] == '1';
            }();
            static int host_dbg_round = 0;
            if (host_dbg && host_dbg_round < 12) {
                ++host_dbg_round;
                std::fprintf(stderr, "[mtp-host] r=%d B=%d k=%u anchor=", host_dbg_round,
                             batch_size, k);
                for (int i = 0; i < batch_size; ++i) {
                    std::fprintf(stderr, "%d ", static_cast<int>(state.host_ingress.anchors[i]));
                }
                std::fprintf(stderr, "extent=");
                for (int i = 0; i < batch_size; ++i) {
                    std::fprintf(stderr, "%d ",
                                 static_cast<int>(state.host_ingress.current_extents[i]));
                }
                std::fprintf(stderr, "drafts=");
                for (int i = 0; i < batch_size * static_cast<int>(k); ++i) {
                    std::fprintf(stderr, "%d ",
                                 static_cast<int>(state.host_ingress.current_drafts[i]));
                }
                std::fprintf(stderr, "frontier=");
                for (int i = 0; i < batch_size; ++i) {
                    std::fprintf(stderr, "%d ",
                                 static_cast<int>(state.host_ingress.base_frontiers[i]));
                }
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache);
        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets            = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents    = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts     = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope        = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows           = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor rope_deltas        = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents       = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids      = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden   = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden          = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden        = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions       = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions  = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns   = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts        = frame.next_drafts.slice(0, 0, batch_size);

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        // TEMP DEBUG: dump MTP round intermediates. Remove after diagnosis.
        const bool mtp_dbg = [] {
            const char* e = std::getenv("NINFER_MTP_DEBUG");
            return e != nullptr && *e != '\0';
        }();
        auto dump_i32 = [&](const char* name, const Tensor& t, int n) {
            std::vector<std::int32_t> host(static_cast<std::size_t>(n), -7);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), t.data,
                                       static_cast<std::size_t>(n) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, state.execution.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));
            std::fprintf(stderr, "[mtp-dbg] %s:", name);
            for (int i = 0; i < n; ++i) { std::fprintf(stderr, " %d", host[i]); }
            std::fprintf(stderr, "\n");
        };
        if (mtp_dbg) {
            dump_i32("ingress.drafts", current_drafts, static_cast<int>(k) * batch_size);
            dump_i32("ingress.extents", current_extents, batch_size);
            dump_i32("verify_ids", verify_ids, width * batch_size);
            dump_i32("cache_positions", target_positions, width * batch_size);
            dump_i32("rope_positions", target_rope, width * batch_size);
            // SamplingConfig: temperature, top_k, top_p, min_p, presence, frequency,
            // seed(lo,hi), token_counts(lo,hi) -> 9 x i32 per row
            {
                std::vector<std::int32_t> cfg_host(9 * static_cast<std::size_t>(batch_size), -7);
                CUDA_CHECK(cudaMemcpyAsync(
                    cfg_host.data(),
                    static_cast<const unsigned char*>(frame.ingress.data) +
                        offsetof(qwen3_5::MtpDecodeIngress, sampling),
                    cfg_host.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                    state.execution.device.stream));
                CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));
                std::fprintf(stderr, "[mtp-dbg] configs.raw:");
                for (std::int32_t v : cfg_host) { std::fprintf(stderr, " %d", v); }
                std::fprintf(stderr, "\n");
            }
        }
        mt_mark(mt.prepare);
        {
            nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                           static_cast<std::uint64_t>(width) * batch_size);
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 TargetVerifyFrameView{
                                     .ids                     = verify_ids,
                                     .cache_positions         = target_positions,
                                     .rope_positions          = target_rope,
                                     .valid_columns           = target_valid,
                                     .kv_table_rows           = text_rows,
                                     .state_source_slots      = state_sources,
                                     .state_destination_slots = state_destinations,
                                     .target_hidden           = target_hidden,
                                     .target_logits           = target_logits,
                                     .target_tokens           = target_tokens,
                                     .drafts                  = current_drafts,
                                     .current_extents         = current_extents,
                                     .frontiers               = frontiers,
                                     .anchors                 = anchors,
                                     .licensed_tokens         = licensed_tokens,
                                     .licensed_counts         = licensed_counts,
                                     .accepted_drafts         = accepted,
                                     .selected_hidden         = selected_hidden,
                                     .replay_records          = state.execution.replay_records,
                                     .sampling                = frame.sampling,
                                 },
                                 envelopes.target_verify);
        }

        if (mtp_dbg) {
            dump_i32("post-verify.target_tokens", target_tokens, width * batch_size);
            dump_i32("post-verify.licensed_counts", licensed_counts, batch_size);
            dump_i32("post-verify.accepted", accepted, batch_size);
            dump_i32("post-verify.anchors", anchors, batch_size);
        }

        mt_mark(mt.verify);
        if (mt.on) {
            // Acceptance is the one number that decides whether a wider draft window is
            // worth its round time, so it is reported alongside the timings.
            std::int32_t host_accepted = 0;
            CUDA_CHECK(cudaMemcpyAsync(&host_accepted, accepted.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, state.execution.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));
            mt.accepted += host_accepted;
        }
        {
            nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                          static_cast<std::uint64_t>(k) * batch_size);
            ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                        licensed_counts, rope_deltas, alignment_ids, next_extents,
                                        ar_positions, ar_rope_positions, ar_valid_columns,
                                        static_cast<std::int32_t>(state.text_cache.max_context()),
                                        state.execution.device.stream);
            if (mtp_dbg) {
                dump_i32("post-prepare.next_extents", next_extents, batch_size);
            }
            card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions,
                                          target_rope, licensed_counts, mtp_rows, envelopes.batch,
                                          alignment_hidden);
            mt_split(mt.batch_fwd);
            ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                    state.execution.device.stream);

            Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
            Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
            if (mtp_dbg) {
                // AMD port diagnosis: the draft AR starts from the aligned hidden. If the
                // MTP head computes a different hidden space than the target model, the
                // shared lm-head proposal is garbage and acceptance collapses. Compare the
                // aligned hidden (ar_hidden) against the target model hidden (target_hidden)
                // at the same position.
                auto dump_hidden = [&](const char* name, const Tensor& t, int n) {
                    std::vector<float> h(static_cast<std::size_t>(n));
                    CUDA_CHECK(cudaMemcpyAsync(
                        h.data(), t.data, static_cast<std::size_t>(n) * sizeof(float),
                        cudaMemcpyDeviceToHost, state.execution.device.stream));
                    CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));
                    double sum = 0.0, sum2 = 0.0, nz = 0.0;
                    int nan = 0;
                    for (float v : h) {
                        sum += v; sum2 += static_cast<double>(v) * v;
                        if (v != 0.0f) nz += 1.0;
                        if (v != v) nan += 1;
                    }
                    std::fprintf(stderr, "[mtp-hidden] %s mean=%.4f rms=%.4f nz=%.1f%% nan=%d first=%f\n",
                                 name, sum / n, n > 0 ? std::sqrt(sum2 / n) : 0.0,
                                 n > 0 ? 100.0 * nz / n : 0.0, nan,
                                 n > 0 ? h[0] : 0.0f);
                };
                // target_hidden is [H, width, batch]; take column 0's H values.
                dump_hidden("ar_hidden", ar_hidden, 64);
                dump_hidden("target_hidden", target_hidden, 64);
            }
            card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
            mt_split(mt.propose);
            for (std::uint32_t step = 0; step + 1 < k; ++step) {
                Tensor previous =
                    next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
                // AMD port diagnostic (NINFER_MTP_TEACHER=1): feed the TARGET model's own
                // greedy tokens into the draft AR chain instead of the draft's proposals.
                // MEASURED otherwise: P(draft1 correct) = 0.94 but P(draft2 | draft1) = 0.39
                // on real text, i.e. the chain loses accuracy the moment it starts feeding
                // itself. If acceptance stays high under teacher forcing, the AR step is
                // fine and the collapse is pure error compounding; if it stays low, the AR
                // step itself is wrong. The AR step's own hidden state still comes from the
                // draft chain, so this is not a full teacher forcing.
                static const bool teacher = [] {
                    const char* e = std::getenv("NINFER_MTP_TEACHER");
                    return e != nullptr && e[0] == '1';
                }();
                if (teacher) {
                    // target_argmax is [width][1][batch]; after slicing dim 1 the row
                    // index lives on dim 0, unlike next_drafts whose row index is dim 1.
                    previous = target_tokens.slice(0, static_cast<std::int32_t>(step) + 1, 1)
                                   .view({batch_size});
                }
                Tensor next =
                    next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
                Tensor position =
                    ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
                Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                                  .view({1, batch_size});
                Tensor valid = ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1)
                                   .view({batch_size});
                Tensor previous_batch = previous.view({1, batch_size});
                Tensor hidden_batch   = ar_hidden.view(
                    {dimension(state.execution.parameters.model.config().text.hidden_size), 1,
                       batch_size});
                Tensor next_hidden_batch = next_hidden.view(
                    {dimension(state.execution.parameters.model.config().text.hidden_size), 1,
                     batch_size});
                card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                              mtp_rows, envelopes.ar[step], next_hidden_batch);
                mt_split(mt.ar_fwd);
                card.mtp_propose_batch(next_hidden, proposal_logits, next);
                mt_split(mt.ar_prop);
                CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           state.execution.device.stream));
                mt_split(mt.ar_copy);
            }
        }

        mt_mark(mt.draft_ar);
        if (mt.on) {
            CUDA_CHECK(cudaDeviceSynchronize());
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            mt.total += static_cast<double>(t.QuadPart - mt_start.QuadPart) * 1000.0 / static_cast<double>(mt.freq.QuadPart);
            if (++mt.rounds % 16 == 0) {
                std::fprintf(stderr, "[mtp-timing] %d rounds: verify=%.1f draft_ar=%.1f prepare=%.1f total=%.1f ms (avg/round=%.1f, accepted/round=%.2f)\n",
                             mt.rounds, mt.verify, mt.draft_ar, mt.prepare, mt.total, mt.total / 16.0,
                             static_cast<double>(mt.accepted) / 16.0);
                std::fprintf(stderr, "[mtp-draft] batch_fwd=%.1f propose0=%.1f ar_fwd=%.1f ar_prop=%.1f ar_copy=%.1f (ms per 16 rounds)\n",
                             mt.batch_fwd, mt.propose, mt.ar_fwd, mt.ar_prop, mt.ar_copy);
                std::fflush(stderr);
                mt.verify = mt.draft_ar = mt.prepare = mt.total = 0.0;
                mt.batch_fwd = mt.propose = mt.ar_fwd = mt.ar_prop = mt.ar_copy = 0.0;
                mt.accepted = 0;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_5::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution
