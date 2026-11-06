#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/kv_window.h"
#include "ops/softmax_attention/dense/causal_cache/key_position_hook.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/workspace.h"
#include "models/qwen3_5/program/storage/kv_retrieval.h"

#include "core/nvtx.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/common/nan_triage_probe.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"

namespace {

// Resident window: hand the decode/prefill kernels an explicit slot -> compact
// position map for the lifetime of one launch region. Raised only around the calls
// that consume it, because the hook is process wide while work_ is a reused arena:
// a stale pointer there would turn unrelated attention calls into dirty reads.
class ScopedKeyPositionMap {
public:
    ScopedKeyPositionMap() = default;
    ScopedKeyPositionMap(ninfer::Tensor* map, std::int32_t slots) { arm(map, slots); }
    void arm(ninfer::Tensor* map, std::int32_t slots) {
        ninfer::ops::g_attention_key_position_map =
            map == nullptr ? nullptr : static_cast<const std::int32_t*>(map->data);
        ninfer::ops::g_attention_key_slots = slots;
    }
    ~ScopedKeyPositionMap() {
        ninfer::ops::g_attention_key_position_map = nullptr;
        ninfer::ops::g_attention_key_slots          = 0;
    }
    ScopedKeyPositionMap(const ScopedKeyPositionMap&)            = delete;
    ScopedKeyPositionMap& operator=(const ScopedKeyPositionMap&) = delete;
};

}  // namespace
#include "ops/launcher/paged_kv_window.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <map> // AMD port: prefill profile aggregation
#include <windows.h> // AMD port: QueryPerformanceCounter decode timer
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

// AMD port debug: NINFER_LOGIT_PROBE=1 prints one logits column's argmax plus the value at a
// reference token, so a verify-vs-AR head A/B is a single stderr line instead of a dump. The
// verify and AR heads are the last place they can disagree once the layer trace is bit-equal.
void debug_logit_probe(const char* who, const Tensor& logits, std::int32_t column,
                       std::int32_t ref_token, cudaStream_t stream) {
    static const char* enabled = std::getenv("NINFER_LOGIT_PROBE");
    if (enabled == nullptr || *enabled == '\0') { return; }
    const std::int64_t vocab = logits.ne[0];
    if (vocab <= 0 || logits.data == nullptr) { return; }
    std::vector<std::uint16_t> host(static_cast<std::size_t>(vocab));
    const std::size_t col_bytes = static_cast<std::size_t>(vocab) * sizeof(std::uint16_t);
    const std::uint8_t* src = static_cast<const std::uint8_t*>(logits.data) +
                              static_cast<std::int64_t>(column) * logits.nb[1];
    CUDA_CHECK(cudaMemcpy2DAsync(host.data(), col_bytes, src, logits.nb[1], col_bytes, 1,
                                 cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float best = -1.0e30f;
    std::int64_t best_i = -1;
    float ref = 0.0f;
    std::int64_t ref_rank = 0;
    // AMD port: the rank of the reference token and the head of the distribution separate
    // "the head is nearly right but numerically off" from "the head computes something else
    // entirely", which a bare argmax comparison cannot.
    std::int64_t top_i[5] = {-1, -1, -1, -1, -1};
    float top_v[5]      = {-1.0e30f, -1.0e30f, -1.0e30f, -1.0e30f, -1.0e30f};
    for (std::int64_t i = 0; i < vocab; ++i) {
        const float v = std::bit_cast<float>(static_cast<std::uint32_t>(host[i]) << 16);
        if (v > best) { best = v; best_i = i; }
        if (i == static_cast<std::int64_t>(ref_token)) { ref = v; }
        if (v > ref) { ++ref_rank; }
        for (int slot = 0; slot < 5; ++slot) {
            if (v > top_v[slot]) {
                for (int k = 4; k > slot; --k) { top_v[k] = top_v[k - 1]; top_i[k] = top_i[k - 1]; }
                top_v[slot] = v;
                top_i[slot] = i;
                break;
            }
        }
    }
    std::fprintf(stderr,
                 "[logit-probe] %s col=%d argmax=%lld val=%.6f ref%d=%.6f rank=%lld "
                 "top5=%lld:%.4f,%lld:%.4f,%lld:%.4f,%lld:%.4f,%lld:%.4f\n",
                 who, static_cast<int>(column), static_cast<long long>(best_i), best, ref_token, ref,
                 static_cast<long long>(ref_rank), static_cast<long long>(top_i[0]), top_v[0],
                 static_cast<long long>(top_i[1]), top_v[1], static_cast<long long>(top_i[2]),
                 top_v[2], static_cast<long long>(top_i[3]), top_v[3],
                 static_cast<long long>(top_i[4]), top_v[4]);
    std::fflush(stderr);
}

// AMD port debug: per-layer hidden-state checksum, enabled by NINFER_DEBUG_LAYERS.
void debug_layer_checksum(const char* tag, int layer, const Tensor& x, cudaStream_t stream) {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_DEBUG_LAYERS");
        return env != nullptr && *env != '\0';
    }();
    if (!enabled) { return; }
    {   // element-count filter (declared early so it guards both print and dump)
        static const long lim = [] {
            const char* e = std::getenv("NINFER_DUMP_MAX_ELEMS");
            return e != nullptr ? std::atol(e) : -1L;
        }();
        if (lim >= 0 && static_cast<long>(x.numel()) > lim) { return; }
    }
    // AMD port: NINFER_DUMP_MIN_T=<T> restricts the whole probe to one tensor width and
    // NINFER_DUMP_MAX=<n> caps the record count, so a verify-vs-decode A/B stays a few MB
    // instead of dumping the entire prefill plus every decode step (measured 8.4 GB once).
    static const int probe_t = [] {
        const char* e = std::getenv("NINFER_DUMP_MIN_T");
        return e != nullptr ? static_cast<int>(std::atoi(e)) : -1;
    }();
    if (probe_t >= 0 && x.ne[1] != probe_t) { return; }
    // AMD port: NINFER_DUMP_MAX_ELEMS skips tensors wider than N elements, which keeps a
    // decode/verify-tail A/B bit-exact without dragging the prefill chunks along.
    static const long probe_max_elems = [] {
        const char* e = std::getenv("NINFER_DUMP_MAX_ELEMS");
        return e != nullptr ? std::atol(e) : -1L;
    }();
    static const long probe_max = [] {
        const char* e = std::getenv("NINFER_DUMP_MAX");
        return e != nullptr ? std::atol(e) : -1L;
    }();
    static long probe_count = 0;
    if (probe_max >= 0 && probe_count >= probe_max) { return; }
    ++probe_count;
    const std::size_t count = static_cast<std::size_t>(x.numel());
    if (count == 0 || x.data == nullptr) { return; }
    std::vector<std::uint16_t> host(count);
    const std::size_t width_bytes = static_cast<std::size_t>(x.ne[0]) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpy2DAsync(host.data(), width_bytes, x.data, x.nb[1], width_bytes,
                                 count / static_cast<std::size_t>(x.ne[0]), cudaMemcpyDeviceToHost,
                                 stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Per-op timeline: NINFER_DEBUG_LAYERS=2 prints ms-since-first-call before each
    // checksum line. The sync above makes each op complete first, so consecutive
    // stamps give the cost of every engine op (coarse but complete coverage).
    static const std::clock_t prof_t0 = std::clock();
    static const bool prof_all = [] {
        const char* e = std::getenv("NINFER_DEBUG_LAYERS");
        return e != nullptr && e[0] == '2';
    }();
    const double prof_ms = prof_all
        ? static_cast<double>(std::clock() - prof_t0) * 1000.0 / CLOCKS_PER_SEC
        : -1.0;
    double sum = 0.0;
    double magnitude = 0.0;
    double minimum = 1.0e30;
    double maximum = -1.0e30;
    std::size_t nonfinite = 0;
    std::size_t nonzero = 0;
    for (const std::uint16_t bits : host) {
        const float value = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
        if (!std::isfinite(value)) { ++nonfinite; continue; }
        if (bits != 0) { ++nonzero; }
        sum += static_cast<double>(value);
        magnitude += std::abs(static_cast<double>(value));
        minimum = std::min(minimum, static_cast<double>(value));
        maximum = std::max(maximum, static_cast<double>(value));
    }
    if (prof_all) {
        std::fprintf(stderr, "[t=%9.1f ms] ", prof_ms);
    }
    std::fprintf(stderr, "[dbg] %s layer=%d T=%d sum=%.6g |v|=%.6g min=%.6g max=%.6g "
                         "nan/inf=%zu zero=%.1f%%\n",
                 tag, layer, x.ne[1], sum, magnitude, minimum, maximum, nonfinite,
                 100.0 * (1.0 - static_cast<double>(nonzero) / static_cast<double>(count)));
    // AMD port debug: raw per-layer last-column dump for prefill-vs-decode A/B.
    static const char* dump_prefix = std::getenv("NINFER_DUMP_LAYERS");
    if (dump_prefix != nullptr && *dump_prefix != '\0') {
        static FILE* dump_file = [prefix = std::string(dump_prefix)] {
            std::string path = prefix + ".bin";
            return fopen(path.c_str(), "wb");
        }();
        if (dump_file != nullptr) {
            const std::int32_t d = static_cast<std::int32_t>(x.ne[0]);
            const std::int32_t t = static_cast<std::int32_t>(count / static_cast<std::size_t>(d));
            std::int32_t header[18] = {};
            for (int i = 0; i < 15 && tag[i] != '\0'; ++i) {
                reinterpret_cast<char*>(header)[i] = tag[i];
            }
            header[15] = d;
            header[16] = layer;
            header[17] = t;
            std::fwrite(header, sizeof(header), 1, dump_file);
            std::fwrite(host.data(), sizeof(std::uint16_t), static_cast<std::size_t>(d) *
                                                              static_cast<std::size_t>(t),
                        dump_file);
            std::fflush(dump_file);
        }
    }
}

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const execution::Parameters& weights,
                         WorkspaceArena& work, qwen3_5::PagedKVCacheView kv,
                         LinearAttentionStatePool& state, qwen3_5::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, qwen3_5::PagedKVCacheView mtp_kv,
                         const qwen3_5::PagedKVCache* batch_text_kv,
                         const qwen3_5::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), parameters_(weights), config_(weights.model.config().text), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io), prefill_hidden_(prefill_hidden),
      prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base), batch_text_kv_(batch_text_kv),
      batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    embed_      = &parameters_.text.token_embedding;
    final_norm_ = &parameters_.text.final_norm;
    lm_head_    = &parameters_.text.output_head;
    mtp_        = parameters_.mtp ? &*parameters_.mtp : nullptr;
    if (mtp_enabled() && mtp_ == nullptr) {
        throw std::invalid_argument("MTP state requires selected MTP parameters");
    }
    if (parameters_.proposal) {
        const auto& p = *parameters_.proposal;
        set_proposal_head(
            &p.head, p.token_ids ? static_cast<const std::int32_t*>(p.token_ids->data) : nullptr,
            dimension(p.rows));
    }
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), T});

    auto roots = workspace::mtp_stem(work_, config_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != dimension(config_.hidden_size) ||
            input_embeddings->numel() !=
                static_cast<std::int64_t>(dimension(config_.hidden_size)) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp_->embedding_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(flat_hidden, mtp_->hidden_norm, config_.rms_norm_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    project(fc_in, mtp_->input_projection, x, work_, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp_->input_norm, config_.rms_norm_eps, true, ah, s);
}

namespace {

// NINFER_MTP_ATTN_WINDOW: keys the MTP draft reads, plus one sink page. Deliberately a file-static
// rather than a TextContext member - text.h is included transitively by every program TU, so adding
// a member here silently invalidates the layout of every object built before the change.
std::uint32_t mtp_attention_window() {
    static const std::uint32_t keys = [] {
        const char* w = std::getenv("NINFER_MTP_ATTN_WINDOW");
        const long v  = (w != nullptr && w[0] != '\0') ? std::strtol(w, nullptr, 10) : 0;
        return v > 0 ? static_cast<std::uint32_t>(v) : 0u;
    }();
    return keys;
}

} // namespace

PagedKVBatchLayerView windowed_mtp_view(WorkspaceArena& work_, DeviceContext& ctx_,
                                        const PagedKVBatchLayerView& base, Tensor& table_rows,
                                        Tensor& positions, const Tensor& valid_columns,
                                        Tensor& window_rows_out, Tensor& window_positions_out,
                                        Tensor& window_tables_out) {
    // The draft reads one sink page plus the newest window, so its attention cost stops growing
    // with the context. Appends keep using the cache's absolute pages; only this read view narrows.
    const int pages  = base.block_tables.ne[0];
    const int rows   = table_rows.ne[0];
    const int width  = positions.ne[0];
    const auto roots = workspace::paged_kv_window(work_, pages, width, rows);
    window_tables_out    = roots.tables;
    window_rows_out      = roots.rows;
    window_positions_out = roots.positions;
    ops::detail::paged_kv_window_rows_launch(
        base.block_tables, table_rows, positions, valid_columns, 1,
        static_cast<std::int32_t>(mtp_attention_window()), window_tables_out, window_rows_out,
        window_positions_out, ctx_.stream);
    PagedKVBatchLayerView narrowed = base;
    narrowed.block_tables          = window_tables_out;
    return narrowed;
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace::mtp_attention_projection(work_, config_, T);
    Tensor q              = projection.query.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_attention_heads), T});
    Tensor k              = projection.key.view({dimension(config_.attention->head_dim),
                                                 dimension(config_.attention->num_key_value_heads), T});
    Tensor gate           = projection.gate.view({dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_attention_heads), T});
    Tensor v              = projection.value.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat         = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat      = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat         = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat         = v.view({dimension(config_.attention->key_width()), T});
    mtp_projection(ah, mtp_->projection, *config_.attention, q_flat, gate_flat, k_flat, v_flat,
                   work_, s);

    const auto results = workspace::mtp_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, s);

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    // AMD DEBUG (MTP attention health): pre-fill the result buffer with a recognisable bf16
    // pattern (0x5A5A, a huge finite value) before the attention call. A zero result then proves
    // the kernel really returned zeros; a huge one proves it never wrote the buffer at all.
    static const bool mtp_prefill_probe = [] {
        const char* e = std::getenv("NINFER_MTP_PREFILL");
        return e != nullptr && e[0] == '1';
    }();
    if (mtp_prefill_probe) {
        const std::size_t p_bytes =
            static_cast<std::size_t>(dimension(config_.attention->head_dim)) *
            static_cast<std::size_t>(dimension(config_.attention->num_attention_heads)) *
            static_cast<std::size_t>(T) * dtype_size(DType::BF16);
        CUDA_CHECK(cudaMemsetAsync(a.data, 0x5A, p_bytes, s));
    }
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        // AMD port: for the MTP draft AR steps (width == 1) the per-step valid_columns is
        // a single 0/1 flag, which pushes this call onto PagedKVBatchMetadata<true>. The
        // main model's decode reaches the same kernel through causal_softmax_attention_cached
        // with a NULL valid_columns and therefore the direct metadata, and that is the only
        // combination the engine has ever validated. For batch == 1 the two resolve the same
        // page table, so passing a null mask here puts the AR step on the proven path.
        // NINFER_MTP_MASKED=1 restores the original routing.
        static const bool keep_masked = [] {
            const char* e = std::getenv("NINFER_MTP_MASKED");
            return e != nullptr && e[0] == '1';
        }();
        const Tensor& ar_valid =
            (width == 1 && !keep_masked) ? Tensor{} : *active_valid_columns_;
        PagedKVBatchLayerView kv_view = batch_mtp_kv_->batch_layer_view(0);
        Tensor table_rows = *active_backend_kv_table_rows_;
        Tensor win_rows{};
        Tensor win_positions{};
        Tensor win_tables{};
        if (mtp_attention_window() != 0) {
            kv_view = windowed_mtp_view(work_, ctx_, kv_view, table_rows, position_batch,
                                        width == 1 ? Tensor{} : *active_valid_columns_, win_rows,
                                        win_positions, win_tables);
            position_batch = win_positions.view({width, active_sequence_batch_});
            table_rows     = win_rows;
        }
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, ar_valid,
            table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            kv_view, envelope, work_, a_batch, s);
    } else {
        // Single-row form: the window still applies, because this is the shape the draft's prefill
        // chunks and the final-column proposal run through.
        PagedKVBatchLayerView kv_view = batch_mtp_kv_->batch_layer_view(0);
        Tensor rows_in                = io_.backend_kv_table_row;
        Tensor pos_rows               = positions.view({T, 1});
        Tensor win_rows{};
        Tensor win_positions{};
        Tensor win_tables{};
        if (mtp_attention_window() != 0) {
            kv_view = windowed_mtp_view(work_, ctx_, kv_view, rows_in, pos_rows, Tensor{}, win_rows,
                                        win_positions, win_tables);
            pos_rows = win_positions.view({T, 1});
            rows_in  = win_rows;
        }
        ops::causal_softmax_attention(
            qn, kn, v, pos_rows.view({T}), Tensor{}, rows_in,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            kv_view, envelope, work_, a, s);
    }
    // AMD DEBUG (MTP attention relevance): the draft of a decode round is computed from the
    // MTP head only. If zeroing the attention output leaves every draft unchanged, the head is
    // running on the embedding/hidden pair alone and every KV-side theory (window size, table
    // rows, cache contents) is moot. NINFER_MTP_ZEROATTN=1 forces that experiment.
    static const bool mtp_zero_attn = [] {
        const char* e = std::getenv("NINFER_MTP_ZEROATTN");
        return e != nullptr && e[0] == '1';
    }();
    if (mtp_zero_attn) {
        const std::size_t a_bytes =
            static_cast<std::size_t>(dimension(config_.attention->head_dim)) *
            static_cast<std::size_t>(dimension(config_.attention->num_attention_heads)) *
            static_cast<std::size_t>(T) * dtype_size(DType::BF16);
        CUDA_CHECK(cudaMemsetAsync(a.data, 0, a_bytes, s));
    }
    static int mtp_tdbg_calls = 0;
    if (std::getenv("NINFER_MTP_TDBG") != nullptr && mtp_tdbg_calls++ < 24) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(s, &cap));
        if (cap == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            const int a_elems = dimension(config_.attention->head_dim) *
                                dimension(config_.attention->num_attention_heads) * T;
            auto norm_of = [&](const Tensor& t, int count) -> double {
                std::vector<std::uint16_t> buf(static_cast<std::size_t>(count));
                CUDA_CHECK(cudaMemcpy(buf.data(), t.data, buf.size() * 2, cudaMemcpyDeviceToHost));
                double acc = 0.0;
                for (auto b : buf) {
                    std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
                    float f         = *reinterpret_cast<float*>(&u);
                    acc += static_cast<double>(f) * static_cast<double>(f);
                }
                return std::sqrt(acc);
            };
            std::fprintf(stderr,
                         "[mtp-tdbg] T=%d batch=%d width=%d env=[%u,%u] ||a||=%.3f ||gate||=%.3f\n", T,
                         active_sequence_batch_, active_sequence_width_,
                         static_cast<unsigned>(envelope.min_visible_keys),
                         static_cast<unsigned>(envelope.max_visible_keys), norm_of(a, a_elems),
                         norm_of(gate, a_elems));
            std::fflush(stderr);
        }
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace::mtp_post_attention(work_, config_, T);
    Tensor o        = post.output;
    project(a.view({dimension(config_.attention->query_width()), T}), mtp_->output, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        ffn(mh, mtp_->ffn, x, {}, work_, s, true);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({dimension(config_.hidden_size), T});
    ops::rmsnorm(x, mtp_->final_norm, config_.rms_norm_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    // AMD port: NINFER_PROFILE_DECODE=1 splits the MTP layer forward into stem and tail.
    // MEASURED before this was added: one MTP-layer forward on ONE token costs 3.4-4.5 ms
    // wall, while the layer's own weights are ~100 MB (0.2 ms at the card's streaming
    // rate). That fixed cost is what the speculative draft AR chain pays k-1 times per
    // round, so it is worth knowing which half of the layer it lives in.
    static const bool mtp_timing = [] {
        const char* e = std::getenv("NINFER_PROFILE_DECODE");
        return e != nullptr && e[0] == '1';
    }();
    LARGE_INTEGER mt0{}, mt1{};
    if (mtp_timing) { QueryPerformanceCounter(&mt0); }
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    if (mtp_timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        QueryPerformanceCounter(&mt1);
    }
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
    if (mtp_timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        LARGE_INTEGER mt2{};
        QueryPerformanceCounter(&mt2);
        static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
        static double s_stem = 0, s_tail = 0;
        static long s_calls = 0;
        s_stem += static_cast<double>(mt1.QuadPart - mt0.QuadPart) * 1000.0 / freq.QuadPart;
        s_tail += static_cast<double>(mt2.QuadPart - mt1.QuadPart) * 1000.0 / freq.QuadPart;
        if (++s_calls % 16 == 0) {
            std::fprintf(stderr, "[mtp-layer] %ld calls: stem=%.1f tail=%.1f ms (per call %.2f / %.2f)\n",
                         s_calls, s_stem, s_tail, s_stem / 16.0, s_tail / 16.0);
            std::fflush(stderr);
            s_stem = s_tail = 0;
        }
    }
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    // AMD DEBUG: the MTP proposal for a prompt is produced by whichever chunk reaches
    // is_last, so the chunk geometry (T, final flag, absolute positions) decides what the
    // draft head actually consumed. Print it when tracing MTP alignment.
    if (std::getenv("NINFER_MTP_TRACE") != nullptr) {
        std::int32_t trace_first = -1;
        std::int32_t trace_last  = -1;
        CUDA_CHECK(cudaMemcpy(&trace_first, positions.data, sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&trace_last,
                              static_cast<const std::int32_t*>(positions.data) + (T - 1),
                              sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "[mtp-trace] T=%d final=%d pos=%d..%d\n", T,
                     final_chunk ? 1 : 0, trace_first, trace_last);
        std::fflush(stderr);
    }
    Tensor x_last;
    Tensor ah_last;
    Tensor kn_last;
    Tensor v_last;
    // AMD DEBUG (MTP head oracle): the MTP stem pairs column i's target hidden with the
    // embedding of the NEXT token, so column i must reproduce prompt token i+2. The last
    // column's expected token is not in the prompt, but column T-1-n must reproduce
    // ids[T-n], which the prompt already contains. NINFER_MTP_COL_BACK=n therefore turns
    // the draft head into a checkable oracle: with n=1 the proposal must equal the token
    // the prefill already sampled (io_.token).
    int mtp_col_back = 0;
    if (final_chunk) {
        if (const char* e = std::getenv("NINFER_MTP_COL_BACK")) { mtp_col_back = std::atoi(e); }
        if (mtp_col_back < 0 || mtp_col_back > T - 2) { mtp_col_back = 0; }
    }
    const int mtp_final_col = T - 1 - mtp_col_back;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ah_last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        kn_last = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                            dimension(config_.attention->num_key_value_heads), 1});
        v_last  = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                            dimension(config_.attention->num_key_value_heads), 1});
    }

    // AMD DEBUG (MTP teacher-forced oracle): keep a tail slab of the stem output so the
    // final-column pass can be replayed after the bulk workspace scope closes, for ANY chunk (the
    // engine only marks the last 4-column warm-up chunk as final, which is too few samples).
    int oracle_cols = 0;
    if (const char* e = std::getenv("NINFER_MTP_ORACLE_LOOP")) { oracle_cols = std::atoi(e); }
    if (oracle_cols < 0) { oracle_cols = 0; }
    if (oracle_cols > T - 2) { oracle_cols = T - 2; }
    if (std::getenv("NINFER_MTP_DIRECT_ATTN") != nullptr &&
        std::getenv("NINFER_MTP_ORACLE_DIRECT") == nullptr) {
        oracle_cols = 0;
    }
    Tensor x_tail;
    Tensor ah_tail;
    if (oracle_cols > 0) {
        x_tail  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), oracle_cols});
        ah_tail = work_.alloc(DType::BF16, {dimension(config_.hidden_size), oracle_cols});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
        if (oracle_cols > 0) {
            const std::size_t oracle_column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const std::size_t first = static_cast<std::size_t>(T - 1 - oracle_cols);
            CUDA_CHECK(cudaMemcpyAsync(x_tail.data,
                                       static_cast<const unsigned char*>(x.data) +
                                           first * oracle_column_bytes,
                                       oracle_column_bytes * static_cast<std::size_t>(oracle_cols),
                                       cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(ah_tail.data,
                                       static_cast<const unsigned char*>(ah.data) +
                                           first * oracle_column_bytes,
                                       oracle_column_bytes * static_cast<std::size_t>(oracle_cols),
                                       cudaMemcpyDeviceToDevice, s));
        }

        Tensor k_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        Tensor v_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        mtp_kv_projection(ah, mtp_->projection, *config_.attention, k_flat, v_flat, work_, s);
        Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor v = v_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor kn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_key_value_heads), T});
        ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
        text_rope(rope_positions, *config_.rope_parameters, kn, s);
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);
        // AMD DEBUG (MTP kv health): a cached read can only return the appended rows, so a zero
        // value plane is indistinguishable from a zero attention output at the layer level.
        static int mtp_kv_dbg_calls = 0;
        if (std::getenv("NINFER_MTP_KVDBG") != nullptr && mtp_kv_dbg_calls++ < 3) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            const int kn_elems = dimension(config_.attention->head_dim) *
                                 dimension(config_.attention->num_key_value_heads) * T;
            auto tnorm = [&](const Tensor& t, int count) -> double {
                std::vector<std::uint16_t> buf(static_cast<std::size_t>(count));
                CUDA_CHECK(
                    cudaMemcpy(buf.data(), t.data, buf.size() * 2, cudaMemcpyDeviceToHost));
                double acc = 0.0;
                int zeros   = 0;
                for (auto b : buf) {
                    std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
                    float f         = *reinterpret_cast<float*>(&u);
                    if (f == 0.0f) { ++zeros; }
                    acc += static_cast<double>(f) * static_cast<double>(f);
                }
                std::fprintf(stderr, " zeros=%d/%d", zeros, count);
                return std::sqrt(acc);
            };
            std::fprintf(stderr, "[mtp-kv-dbg] T=%d ||kn||=%.3f", T, tnorm(kn, kn_elems));
            std::fprintf(stderr, " ||v||=%.3f", tnorm(v, kn_elems));
            std::fprintf(stderr, " ||ah||=%.3f", tnorm(ah, dimension(config_.hidden_size) * T));
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(mtp_final_col) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(mtp_final_col) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            const std::size_t kv_column_bytes =
                static_cast<std::size_t>(dimension(config_.attention->head_dim)) *
                static_cast<std::size_t>(dimension(config_.attention->num_key_value_heads)) *
                dtype_size(DType::BF16);
            const auto* kn_src = static_cast<const unsigned char*>(kn.data) +
                                 static_cast<std::size_t>(mtp_final_col) * kv_column_bytes;
            const auto* v_src = static_cast<const unsigned char*>(v.data) +
                                static_cast<std::size_t>(mtp_final_col) * kv_column_bytes;
            CUDA_CHECK(cudaMemcpyAsync(kn_last.data, kn_src, kv_column_bytes,
                                       cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(v_last.data, v_src, kv_column_bytes,
                                       cudaMemcpyDeviceToDevice, s));
        }
    }

    // AMD DEBUG (MTP teacher-forced oracle loop): the final-column pass is a pure function of the
    // column index, because the bulk stage already appended every column's K/V. Exposing it as a
    // lambda lets NINFER_MTP_ORACLE_LOOP replay columns T-2, T-3, ... so the draft head can be
    // scored against the prompt's own tokens (a real acceptance number instead of one single
    // judgement call).
    auto run_final_column = [&](int final_col, Tensor x_col, Tensor ah_col, Tensor& hidden_out,
                                Tensor& logits_out, Tensor& token_out) {
        auto col_scope = work_.scope();
        Tensor q_flat = work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor gate_flat =
            work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor q    = q_flat.view({dimension(config_.attention->head_dim),
                                   dimension(config_.attention->num_attention_heads), 1});
        Tensor gate = gate_flat.view({dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor last_position = positions.slice(0, final_col, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, final_col, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src_p = static_cast<const std::int32_t*>(rope_positions.data) +
                                    static_cast<std::size_t>(axis) * T + final_col;
                auto* dst_p = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(cudaMemcpyAsync(dst_p, src_p, sizeof(std::int32_t),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }
        // AMD DEBUG (MTP head oracle packed A/B): the working decode tail computes q/k/gate/v in
        // ONE packed mtp_projection call, while prefill-final historically used the per-row
        // linears (mtp_kv_projection / mtp_query_gate_projection). Both read the same gguf
        // weight slices, but the two linear paths differ (packed+split vs per-row). With
        // NINFER_MTP_PACKED_FINAL=1 the final column mirrors the decode tail exactly: packed
        // projection, rmsnorm+rope for both q and k, then the direct attention variant with the
        // freshly computed normalized k / raw v instead of the bulk-appended cache row.
        const bool packed_final = std::getenv("NINFER_MTP_PACKED_FINAL") != nullptr;
        Tensor qn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor kn_packed;
        Tensor v_packed;
        if (packed_final) {
            Tensor k_flat =
                work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), 1});
            Tensor v_flat =
                work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), 1});
            mtp_projection(ah_col, mtp_->projection, *config_.attention, q_flat, gate_flat,
                           k_flat, v_flat, work_, s);
            Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                    dimension(config_.attention->num_key_value_heads), 1});
            v_packed = v_flat.view({dimension(config_.attention->head_dim),
                                    dimension(config_.attention->num_key_value_heads), 1});
            ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
            kn_packed = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_key_value_heads),
                                                  1});
            ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn_packed, s);
            text_rope(last_rope_position, *config_.rope_parameters, qn, kn_packed, s);
        } else {
            mtp_query_gate_projection(ah_col, mtp_->projection, *config_.attention, q_flat,
                                      gate_flat, work_, s);
            ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
        }
        if (!packed_final) {
            text_rope(last_rope_position, *config_.rope_parameters, qn, s);
        }

        Tensor a = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), 1});
        const auto attn_scale = static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim)));
        // AMD DEBUG (MTP head oracle A/B): the working decode tail uses the append (direct)
        // variant with explicit current-column k/v; prefill-final historically used the cached
        // variant that reads its own row from the cache. NINFER_MTP_DIRECT_ATTN=1 forces the
        // direct path (same call shape as text.cpp decode tail) to isolate which one is wrong.
        if (packed_final) {
            ops::causal_softmax_attention(
                qn, kn_packed, v_packed, last_position, Tensor{}, io_.backend_kv_table_row,
                {dimension(config_.attention->head_dim),
                 dimension(config_.attention->num_attention_heads),
                 dimension(config_.attention->num_key_value_heads)},
                attn_scale, batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
        } else if (std::getenv("NINFER_MTP_DIRECT_ATTN") != nullptr) {
            ops::causal_softmax_attention(
                qn, kn_last, v_last, last_position, Tensor{}, io_.backend_kv_table_row,
                {dimension(config_.attention->head_dim),
                 dimension(config_.attention->num_attention_heads),
                 dimension(config_.attention->num_key_value_heads)},
                attn_scale, batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
        } else {
            ops::causal_softmax_attention_cached(
                qn, last_position,
                {dimension(config_.attention->head_dim),
                 dimension(config_.attention->num_attention_heads),
                 dimension(config_.attention->num_key_value_heads)},
                attn_scale, mtp_kv_.layer_view(0), envelope, work_, a, s);
        }
        // AMD DEBUG (MTP layer ablation): drop the attention branch's contribution to see whether
        // it carries any signal at all. A head that scores the same either way is not reading a
        // useful context, which is a different bug from a weak head.
        const bool skip_mtp_attn = std::getenv("NINFER_MTP_SKIP_ATTN") != nullptr;
        // AMD DEBUG (MTP attention health): a gate that saturates to zero and a KV read that
        // returns zeros both make the branch a no-op, and they need different fixes. Print the raw
        // magnitudes before the gate multiplies them.
        static int mtp_attn_dbg_calls = 0;
        if (std::getenv("NINFER_MTP_ATTN_DBG") != nullptr && mtp_attn_dbg_calls++ < 4) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            auto host_norm = [&](const Tensor& t, int count, bool sigmoid) -> double {
                std::vector<std::uint16_t> buf(static_cast<std::size_t>(count));
                CUDA_CHECK(
                    cudaMemcpy(buf.data(), t.data, buf.size() * 2, cudaMemcpyDeviceToHost));
                double acc = 0.0;
                for (auto b : buf) {
                    std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
                    float f         = *reinterpret_cast<float*>(&u);
                    if (sigmoid) { f = 1.0f / (1.0f + std::exp(-f)); }
                    acc += static_cast<double>(f) * static_cast<double>(f);
                }
                return std::sqrt(acc);
            };
            const int attn_elems = dimension(config_.attention->head_dim) *
                                   dimension(config_.attention->num_attention_heads);
            std::fprintf(stderr,
                         "[mtp-attn-dbg] T=%d col=%d ||a||=%.3f ||gate||=%.3f ||sig(gate)||=%.3f "
                         "||sig(gate)||/sqrt(n)=%.4f ||x||=%.3f\n",
                         T, final_col,
                         host_norm(a, attn_elems, false), host_norm(gate, attn_elems, false),
                         host_norm(gate, attn_elems, true),
                         host_norm(gate, attn_elems, true) / std::sqrt(double(attn_elems)),
                         host_norm(x_col, dimension(config_.hidden_size), false));
            std::fflush(stderr);
        }
        if (!skip_mtp_attn) {
            ops::sigmoid_mul(gate, a, s);

            Tensor o = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
            project(a.view({dimension(config_.attention->query_width()), 1}), mtp_->output, o,
                    work_, s);
            ops::residual_add(o, x_col, s);
        }

        Tensor mh = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ops::rmsnorm(x_col, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);
        if (std::getenv("NINFER_MTP_SKIP_MLP") == nullptr) {
            auto post_mixer_scope = work_.scope();
            ffn(mh, mtp_->ffn, x_col, {}, work_, s, true);
        }
        ops::rmsnorm(x_col, mtp_->final_norm, config_.rms_norm_eps, true, hidden_out, s);
        proposal_argmax(hidden_out, logits_out, token_out);
        };
        if (final_chunk) {
            run_final_column(mtp_final_col, x_last, ah_last, *final_hidden, *logits, *draft_token);
        }

        // AMD DEBUG (MTP teacher-forced oracle loop): replay the final column at T-2, T-3, ... so
        // the draft head is scored against tokens the prompt already contains. A head that is
        // merely mediocre should land near the documented teacher-forced ~38%; a broken one stays
        // at a few percent.
        // AMD DEBUG (target head control): score the TARGET model's own greedy argmax on the very
        // same columns, using the same output head. Without this number there is no way to tell a
        // weak draft head from text the model itself cannot predict.
        const bool target_oracle = std::getenv("NINFER_MTP_TARGET_ORACLE") != nullptr;
        // AMD DEBUG (cross-head oracle): run the full MTP layer on the real column, then score its
        // hidden with the TARGET lm_head instead of mtp_->output_head. This splits the 47.7%
        // draft/target retention (measured on the ASCII corpus 2026-10-29) into two very different
        // causes: a lossy MTP stack (cross-head also low) versus a wrong MTP output head
        // (cross-head lands near the target rate while the mtp-head number stays low).
        const bool xhead_oracle = std::getenv("NINFER_MTP_ORACLE_XHEAD") != nullptr;
        static int scan_hits[5] = {0, 0, 0, 0, 0};
        static int scan_total   = 0;
        if (oracle_cols > 0) {
            int hits = 0;
            for (int back = 1; back <= oracle_cols; ++back) {
                const int col = T - 1 - back;
                const int slab = oracle_cols - back;
                auto loop_scope = work_.scope();
                Tensor hidden_t = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                Tensor logits_t = work_.alloc(DType::BF16, {dimension(config_.vocab_size), 1});
                Tensor token_t  = work_.alloc(DType::I32, {1});
                if (target_oracle) {
                    // The target head, not mtp_->output_head: the MTP head is trained in the MTP
                    // hidden space, so feeding it target hiddens scores like noise.
                    Tensor target_logits = work_.alloc(DType::BF16, {dimension(config_.vocab_size), 1});
                    project(hidden.slice(1, col, 1), *lm_head_, target_logits, work_, s);
                    ops::argmax(target_logits, token_t,
                                dimension(parameters_.model.resources().public_token_count), s);
                } else {
                    run_final_column(col, x_tail.slice(1, slab, 1), ah_tail.slice(1, slab, 1),
                                     hidden_t, logits_t, token_t);
                    if (xhead_oracle) {
                        Tensor xlogits = work_.alloc(DType::BF16, {dimension(config_.vocab_size), 1});
                        project(hidden_t, *lm_head_, xlogits, work_, s);
                        ops::argmax(xlogits, token_t,
                                    dimension(parameters_.model.resources().public_token_count), s);
                    }
                }
                std::int32_t got = -1;
                std::int32_t want = -1;
                std::int32_t emb = -1;
                CUDA_CHECK(cudaStreamSynchronize(s));
                CUDA_CHECK(cudaMemcpy(&got, token_t.data, sizeof(std::int32_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&emb,
                                      static_cast<const char*>(ids.data) +
                                          static_cast<std::size_t>(col) * sizeof(std::int32_t),
                                      sizeof(std::int32_t), cudaMemcpyDeviceToHost));
                // The target head at column j predicts the token at j (ids is the alignment
                // sequence, so ids[j] is already the "next" token); the draft head, which also
                // consumes ids[j] as its embedding, must reach one step further, ids[j + 1].
                const int want_col = target_oracle ? col : col + 1;
                if (want_col < T) {
                    CUDA_CHECK(cudaMemcpy(&want,
                                          static_cast<const char*>(ids.data) +
                                              static_cast<std::size_t>(want_col) *
                                                  sizeof(std::int32_t),
                                          sizeof(std::int32_t), cudaMemcpyDeviceToHost));
                }
                hits += (got == want) ? 1 : 0;
                // AMD DEBUG (offset scan): decide which alignment token the draft pair actually
                // predicts, by scoring the SAME teacher-forced column against ids[col+off] for
                // off = -1..+3. One run yields the full confusion, no pipeline changes.
                if (std::getenv("NINFER_MTP_ORACLE_OFFSCAN") != nullptr) {
                    for (int off = -1; off <= 3; ++off) {
                        const int scan_col = col + off;
                        if (scan_col < 0 || scan_col >= T) { continue; }
                        std::int32_t cand = -1;
                        CUDA_CHECK(cudaMemcpy(&cand,
                                              static_cast<const char*>(ids.data) +
                                                  static_cast<std::size_t>(scan_col) *
                                                      sizeof(std::int32_t),
                                              sizeof(std::int32_t), cudaMemcpyDeviceToHost));
                        if (cand == got) { ++scan_hits[off + 1]; }
                    }
                    ++scan_total;
                }
                std::fprintf(stderr, "[mtp-oracle-loop] %s col=%d embed=%d got=%d want=%d %s\n",
                             target_oracle ? "target" : (xhead_oracle ? "xhead" : "draft"), col, emb,
                             got, want,
                             got == want ? "HIT" : "MISS");
            }
            std::fprintf(stderr, "[mtp-oracle-loop] teacher-forced acceptance %d/%d = %.1f%%\n", hits,
                         oracle_cols, 100.0 * static_cast<double>(hits) / oracle_cols);
            if (std::getenv("NINFER_MTP_ORACLE_OFFSCAN") != nullptr) {
                std::fprintf(stderr,
                             "[mtp-offscan] n=%d off-1=%d off0=%d off+1=%d off+2=%d off+3=%d\n",
                             scan_total, scan_hits[0], scan_hits[1], scan_hits[2], scan_hits[3],
                             scan_hits[4]);
            }
            std::fflush(stderr);
        }

        if (final_chunk && mtp_col_back != 0) {
            std::int32_t oracle_token = -1;
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaMemcpy(&oracle_token, draft_token->data, sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost));
            // AMD port diagnostic: report the pairing inputs and the ground truth so the
            // oracle can be judged without guessing. Column j consumes hidden column j and
            // embedding ids[j]; it predicts the token at prompt position (begin + j + 2),
            // which is ids[j + 1] inside the chunk and therefore directly comparable.
            std::int32_t pair_embed = -1;
            std::int32_t expect     = -1;
            CUDA_CHECK(cudaMemcpy(&pair_embed,
                                  static_cast<const char*>(ids.data) +
                                      static_cast<std::size_t>(mtp_final_col) * sizeof(std::int32_t),
                                  sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            if (mtp_final_col + 1 < T) {
                CUDA_CHECK(cudaMemcpy(
                    &expect,
                    static_cast<const char*>(ids.data) +
                        static_cast<std::size_t>(mtp_final_col + 1) * sizeof(std::int32_t),
                    sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            }
            std::fprintf(stderr,
                         "[mtp-oracle] col_back=%d final_col=%d/T=%d embed=%d proposal=%d "
                         "expected=%d %s\n",
                         mtp_col_back, mtp_final_col, T, pair_embed, oracle_token, expect,
                         expect >= 0 ? (oracle_token == expect ? "HIT" : "MISS") : "(out-of-chunk)");
            std::fflush(stderr);
        }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    auto proposal_scope = work_.scope();
    const int T         = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, dimension(config_.vocab_size), T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens,
                    proposal_head_ids_
                        ? proposal_head_n_
                        : dimension(parameters_.model.resources().public_token_count),
                    ctx_.stream);
        if (proposal_head_ids_ != nullptr) {
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
        // AMD port: NINFER_LOGIT_PROBE + NINFER_MTP_REF=<correct token> reports where the
        // correct token ranks in the MTP head's own output. Enabled only by that env pair.
        static const int mtp_ref = [] {
            const char* e = std::getenv("NINFER_MTP_REF");
            return e != nullptr ? std::atoi(e) : 0;
        }();
        if (mtp_ref > 0) {
            debug_logit_probe("MTP", output_logits, 0, mtp_ref, ctx_.stream);
        }
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    // AMD port: NINFER_PROFILE_DECODE=1 -> per-section host timing with stream syncs
    // (2 syncs per step; at ~40 ms/step the ~20 us overhead is negligible).
    struct DecTimer {
        bool on = false;
        LARGE_INTEGER t0 = {};
        LARGE_INTEGER freq = {};
        double layers_ms = 0.0, head_ms = 0.0, step_ms = 0.0;
        int steps = 0;
        double elapsed_ms(const LARGE_INTEGER& a, const LARGE_INTEGER& b) const {
            return static_cast<double>(b.QuadPart - a.QuadPart) * 1000.0 /
                   static_cast<double>(freq.QuadPart);
        }
    };
    static DecTimer dtimer = [] {
        DecTimer d;
        QueryPerformanceFrequency(&d.freq);
        const char* e = std::getenv("NINFER_PROFILE_DECODE");
        d.on = e != nullptr && e[0] == '1';
        return d;
    }();
    const bool dtiming = dtimer.on && batch == 1;
    if (dtiming) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        QueryPerformanceCounter(&dtimer.t0);
    }
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        ops::embedding(ids, *embed_, x, stream);
        if (dtiming) {
            CUDA_CHECK(cudaDeviceSynchronize());
            LARGE_INTEGER t1;
            QueryPerformanceCounter(&t1);
            dtimer.layers_ms += dtimer.elapsed_ms(dtimer.t0, t1);
            dtimer.t0 = t1;
        }
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden, stream);
        debug_layer_checksum("final-dec", 99, hidden, stream);
        if (dtiming) {
            CUDA_CHECK(cudaDeviceSynchronize());
            LARGE_INTEGER t2;
            QueryPerformanceCounter(&t2);
            dtimer.head_ms += dtimer.elapsed_ms(dtimer.t0, t2);
            dtimer.t0 = t2;
        }
        project(hidden, *lm_head_, logits, work_, stream);
        debug_logit_probe("AR", logits, 0, 5125, stream);
    }
    work_.reset();
    if (dtiming) {
        CUDA_CHECK(cudaDeviceSynchronize());
        LARGE_INTEGER t3;
        QueryPerformanceCounter(&t3);
        dtimer.step_ms += dtimer.elapsed_ms(dtimer.t0, t3);
        if (++dtimer.steps % 64 == 0) {
            std::fprintf(stderr, "[prof-decode] %d steps: layers %.1f ms/step, "
                                 "norm+lmhead %.1f ms/step, step-total %.1f ms/step\n",
                         dtimer.steps, dtimer.layers_ms / 64.0, dtimer.head_ms / 64.0,
                         dtimer.step_ms / 64.0);
            std::fflush(stderr);
            dtimer.layers_ms = dtimer.head_ms = dtimer.step_ms = 0.0;
        }
    }
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        // AMD port: NINFER_PROFILE_VERIFY=1 splits the verify pass into layers / lm_head /
        // argmax with a sync per section. MEASURED WITHOUT THIS: mtp-timing reports the
        // whole verify at ~72 ms while the layer timeline only accounts for ~42 ms, so
        // ~30 ms per round was unattributed.
        static const bool verify_split = [] {
            const char* e = std::getenv("NINFER_PROFILE_VERIFY");
            return e != nullptr && e[0] == '1';
        }();
        struct VerifySplitTimer {
            bool on;
            double layers = 0.0, head = 0.0, arg = 0.0;
            long calls = 0;
            LARGE_INTEGER f = {}, t0 = {}, t1 = {}, t2 = {}, t3 = {};
            void sync_mark(double& acc) {
                CUDA_CHECK(cudaDeviceSynchronize());
                LARGE_INTEGER now;
                QueryPerformanceCounter(&now);
                acc += static_cast<double>(now.QuadPart - t0.QuadPart) * 1000.0 /
                       static_cast<double>(f.QuadPart);
                t0 = now;
            }
        };
        static VerifySplitTimer vs = [] {
            VerifySplitTimer v;
            v.on = verify_split;
            QueryPerformanceFrequency(&v.f);
            return v;
        }();
        if (vs.on) { QueryPerformanceCounter(&vs.t0); }
        run_layers(x, Phase::Verify, tap);
        if (vs.on) { vs.sync_mark(vs.layers); }
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), columns});
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, flat_hidden, stream);
        project(flat_hidden, *lm_head_, flat_logits, work_, stream);
        debug_logit_probe("VF", flat_logits, 0, 5125, stream);
        debug_logit_probe("VF", flat_logits, 1, 1608, stream);
        if (vs.on) { vs.sync_mark(vs.head); }
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
        if (vs.on) {
            vs.sync_mark(vs.arg);
            if (++vs.calls % 16 == 0) {
                std::fprintf(stderr,
                             "[verify-split] %ld calls: layers=%.1f lm_head=%.1f argmax=%.1f ms "
                             "(per call %.2f / %.2f / %.2f)\n",
                             vs.calls, vs.layers, vs.head, vs.arg, vs.layers / 16.0,
                             vs.head / 16.0, vs.arg / 16.0);
                std::fflush(stderr);
                vs.layers = vs.head = vs.arg = 0.0;
            }
        }
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);

    // AMD port diagnostic (NINFER_MTP_XCHECK=1): recompute column 0 through the SINGLE-ROW
    // path (the shape run_final_column uses, which the prefill oracle scores at 55%) and print
    // both argmaxes. If the two disagree the decode batch path is broken; if they agree the
    // head itself is what differs from the target model. The single-row append lands on the
    // same position, so the KV rows stay consistent.
    static const bool mtp_xcheck = [] {
        const char* e = std::getenv("NINFER_MTP_XCHECK");
        return e != nullptr && e[0] == '1';
    }();
    static int xcheck_calls = 0;
    if (mtp_xcheck && width >= 2 && batch == 1 && xcheck_calls < 24) {
        ++xcheck_calls;
        const std::int32_t H = dimension(config_.hidden_size);
        const std::int32_t V = dimension(config_.vocab_size);
        Tensor ids0  = ids.slice(0, 0, 1).view({1});
        Tensor hid0  = hidden.slice(1, 0, 1).view({H, 1});
        Tensor pos0  = cache_positions.slice(0, 0, 1).view({1});
        Tensor rope0 = rope_positions.slice(0, 0, 1).view({1});
        Tensor alt        = workspace::matrix(work_, DType::BF16, H, 1);
        Tensor alt_logits = workspace::matrix(work_, DType::BF16, V, 1);
        Tensor alt_tok    = workspace::vector(work_, DType::I32, 1);
        Tensor base_logits = workspace::matrix(work_, DType::BF16, V, 1);
        Tensor base_tok    = workspace::vector(work_, DType::I32, 1);
        {
            ScopedValue<std::int32_t> xb(active_sequence_batch_, 0);
            ScopedValue<std::int32_t> xw(active_sequence_width_, 0);
            mtp_forward_core(ids0, hid0, pos0, rope0, envelope, alt, nullptr);
        }
        proposal_argmax(alt, alt_logits, alt_tok);
        Tensor col0 = mtp_hidden.slice(1, 0, 1).view({H, 1});
        proposal_argmax(col0, base_logits, base_tok);
        std::int32_t hv_base = -1;
        std::int32_t hv_alt  = -1;
        std::int32_t hpos    = -1;
        std::int32_t ids_h   = -1;
        CUDA_CHECK(cudaMemcpyAsync(&hv_base, base_tok.data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, ctx_.stream));
        CUDA_CHECK(cudaMemcpyAsync(&hv_alt, alt_tok.data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, ctx_.stream));
        CUDA_CHECK(cudaMemcpyAsync(&hpos, pos0.data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, ctx_.stream));
        CUDA_CHECK(cudaMemcpyAsync(&ids_h, ids0.data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, ctx_.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));
        std::fprintf(stderr, "[mtp-xcheck] call=%d pos=%d embed=%d batch_col0=%d single_row=%d\n",
                     xcheck_calls, hpos, ids_h, hv_base, hv_alt);
        std::fflush(stderr);
    }
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const BlockParameters& w, Tensor& x, int fidx, Phase ph) {
    const auto& p  = std::get<AttentionParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }
    debug_layer_checksum("attn-in", fidx, x, s);

    const auto projection = workspace::text_attention_projection(work_, config_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, w.input_norm, config_.rms_norm_eps, true, h, s);

    Tensor q         = projection.query.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_attention_heads), T});
    Tensor gate      = projection.gate.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), T});
    Tensor k         = projection.key.view({dimension(config_.attention->head_dim),
                                            dimension(config_.attention->num_key_value_heads), T});
    Tensor v         = projection.value.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat    = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat    = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat    = v.view({dimension(config_.attention->key_width()), T});
    debug_layer_checksum("h-norm", 0, h, s);
    attention_projection(h, p, q_flat, gate_flat, k_flat, v_flat, work_, s);
    debug_layer_checksum("q-proj", 0, q_flat, s);
    debug_layer_checksum("gate-proj", 0, gate_flat, s);
    debug_layer_checksum("k-proj", 0, k_flat, s);
    debug_layer_checksum("v-proj", 0, v_flat, s);
    if (std::getenv("NINFER_ATTN_PROBE") != nullptr) {
        static int attn_qraw_dump = 0;
        if (attn_qraw_dump == 0) {
            ++attn_qraw_dump;
            const std::size_t qn2 = static_cast<std::size_t>(q_flat.ne[0]) * q_flat.ne[1];
            std::vector<std::uint16_t> qv2(qn2);
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaMemcpy(qv2.data(), q_flat.data, qn2 * 2, cudaMemcpyDeviceToHost));
            std::FILE* qf2 = std::fopen("D:/ninfer-hip-migrate/build/attn_q_raw.bin", "wb");
            if (qf2) { std::fwrite(qv2.data(), 2, qn2, qf2); std::fclose(qf2); }
            std::fprintf(stderr, "[attnprobe] q_raw dumped %zu\n", qn2);
            std::fflush(stderr);
        }
    }

    const auto results = workspace::text_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, p.query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, p.key_norm, config_.rms_norm_eps, true, kn, s);
    debug_layer_checksum("q-norm", fidx, qn, s);
    debug_layer_checksum("k-norm", fidx, kn, s);
    if (std::getenv("NINFER_ATTN_PROBE") != nullptr) {
        // AMD port debug: dump the FIRST attention call of each phase (prefill / decode).
        // Both pick the same compact layer 0 (real layer 3), so the offline CPU reference
        // can use _pre's kn/v as cache row 0 and _dec's qn/kn/v as the decode row.
        static int attn_pre_dumped = 0;
        static int attn_dec_dumped = 0;
        const bool take_norope = (ph == Phase::Prefill && attn_pre_dumped == 0) ||
                                 (ph == Phase::Verify && attn_dec_dumped == 0);
        if (take_norope) {
            if (ph == Phase::Prefill) { ++attn_pre_dumped; } else { ++attn_dec_dumped; }
            const char* pf        = ph == Phase::Prefill ? "pre" : "dec";
            const std::size_t qn2 = static_cast<std::size_t>(qn.ne[0]) * qn.ne[1] * qn.ne[2];
            std::vector<std::uint16_t> qv2(qn2);
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaMemcpy(qv2.data(), qn.data, qn2 * 2, cudaMemcpyDeviceToHost));
            std::string qp = "D:/ninfer-hip-migrate/build/attn_qn_norope_"; qp += pf; qp += ".bin";
            std::FILE* qf2 = std::fopen(qp.c_str(), "wb");
            if (qf2) { std::fwrite(qv2.data(), 2, qn2, qf2); std::fclose(qf2); }
            const std::size_t hn = static_cast<std::size_t>(h.ne[0]) * h.ne[1];
            std::vector<std::uint16_t> hv2(hn);
            CUDA_CHECK(cudaMemcpy(hv2.data(), h.data, hn * 2, cudaMemcpyDeviceToHost));
            std::string hp = "D:/ninfer-hip-migrate/build/attn_h_"; hp += pf; hp += ".bin";
            std::FILE* hf2 = std::fopen(hp.c_str(), "wb");
            if (hf2) { std::fwrite(hv2.data(), 2, hn, hf2); std::fclose(hf2); }
            std::fprintf(stderr, "[attnprobe] %s fi=%d norope qn=%zu h=%zu\n",
                         ph == Phase::Prefill ? "PREFILL" : "DECODE", fidx, qn2, hn);
            std::fflush(stderr);
        }
    }
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    // W4 retrieval: fold this chunk's pre-RoPE K/Q into the block index. The compact layer 0
    // call is the first full-attention layer of every chunk, so the position copy behind
    // begin_chunk happens once per chunk rather than once per layer.
    if (auto* retrieval = ninfer::models::qwen3_5::detail::KVRetrieval::instance();
        retrieval != nullptr) {
        if (fidx == 0) {
            retrieval->begin_chunk(static_cast<const std::int32_t*>(rope_positions.data),
                                   static_cast<std::int32_t>(T), s);
            retrieval->reset_query(s);
        }
    }
    // Resident window: attention kernels derive their scan bound and causal mask from these
    // positions, so they must address the COMPACT KV slot space (absolute - window base).
    // RoPE keeps absolute positions (rope_positions) because the rotation is applied on write.
    Tensor cache_positions_compact;
    const Tensor* cache_positions_ptr = &cache_positions;
    if (g_ninfer_kv_window_base_tokens != 0) {
        const std::int32_t ccount = static_cast<std::int32_t>(cache_positions.ne[0]) *
                                    cache_positions.ne[1] * cache_positions.ne[2] *
                                    cache_positions.ne[3];
        if (ccount > 0) {
            Tensor delta = work_.alloc(DType::I32, {1, 1, 1, 1});
            ops::set_i32_scalar(delta,
                                -static_cast<std::int32_t>(g_ninfer_kv_window_base_tokens), s);
            Tensor target = work_.alloc(DType::I32, {ccount});
            ops::offset_i32_positions(cache_positions.view({ccount}), delta, target, s);
            cache_positions_compact = target.view({cache_positions.ne[0], cache_positions.ne[1],
                                                   cache_positions.ne[2], cache_positions.ne[3]});
            cache_positions_ptr = &cache_positions_compact;
            static std::uint32_t compact_traced_base = 0xFFFFFFFFu;
            if (compact_traced_base != g_ninfer_kv_window_base_tokens) {
                compact_traced_base = g_ninfer_kv_window_base_tokens;
                std::fprintf(stderr, "[win] compact positions: count=%d window_base=%u\n", ccount,
                             static_cast<unsigned>(g_ninfer_kv_window_base_tokens));
                std::fflush(stderr);
            }
        }
    }
    // Paired pre/post-rope dump: hold the pre-rope qn copy in a static host buffer and
    // resolve it against the post-rope qn right after text_rope, so both come from the
    // very same attention call (rotation is norm-preserving, so a norm mismatch means the
    // two dumps landed on different decode steps).
    static std::vector<std::uint16_t> s_rope_pre_qn;
    static std::vector<std::uint16_t> s_rope_pre_kn;
    static int s_rope_plan_pre = 0;
    static int s_rope_plan_dec = 0;
    if (std::getenv("NINFER_ATTN_PROBE") != nullptr) {
        const bool take_pre = (ph == Phase::Prefill && s_rope_plan_pre == 0) ||
                              (ph == Phase::Verify && s_rope_plan_dec == 0);
        if (take_pre) {
            if (ph == Phase::Prefill) { ++s_rope_plan_pre; } else { ++s_rope_plan_dec; }
            const char* pf = ph == Phase::Prefill ? "pre" : "dec";
            const std::int32_t* rp = static_cast<const std::int32_t*>(rope_for_op.data);
            const std::size_t rn = static_cast<std::size_t>(rope_for_op.ne[0]) *
                                   rope_for_op.ne[1] * rope_for_op.ne[2];
            std::vector<std::int32_t> rv(rn > 0 ? rn : 1);
            CUDA_CHECK(cudaStreamSynchronize(s));
            if (rn > 0)
                CUDA_CHECK(cudaMemcpy(rv.data(), rp, rn * 4, cudaMemcpyDeviceToHost));
            std::string rp2 = "D:/ninfer-hip-migrate/build/attn_rope_pos_"; rp2 += pf;
            rp2 += ".bin";
            std::FILE* rf = std::fopen(rp2.c_str(), "wb");
            if (rf) { std::fwrite(rv.data(), 4, rv.size(), rf); std::fclose(rf); }
            std::fprintf(stderr, "[ropeprobe] %s pos ne0=%d ne1=%d rn=%zu v=%d,%d,..\n",
                         ph == Phase::Prefill ? "PREFILL" : "DECODE", (int)rope_for_op.ne[0],
                         (int)rope_for_op.ne[1], rn, rv.empty() ? -1 : rv[0],
                         rv.size() > 1 ? rv[1] : -1);
            std::fflush(stderr);
            const std::size_t qn2 = static_cast<std::size_t>(qn.ne[0]) * qn.ne[1] * qn.ne[2];
            s_rope_pre_qn.assign(qn2, 0);
            CUDA_CHECK(cudaMemcpy(s_rope_pre_qn.data(), qn.data, qn2 * 2, cudaMemcpyDeviceToHost));
            const std::size_t kn2 = static_cast<std::size_t>(kn.ne[0]) * kn.ne[1] * kn.ne[2];
            s_rope_pre_kn.assign(kn2, 0);
            CUDA_CHECK(cudaMemcpy(s_rope_pre_kn.data(), kn.data, kn2 * 2, cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "[ropeprobe] %s pre copy qn2=%zu kn2=%zu\n",
                         ph == Phase::Prefill ? "PREFILL" : "DECODE", qn2, kn2);
            std::fflush(stderr);
        }
    }
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, s);
    debug_layer_checksum("q-rope", fidx, qn, s);
    // Post-rope: qn/kn now hold exactly what the cache stores, so a page mean of these keys
    // dotted with the rotated query equals the mean true attention logit of that page.
    if (auto* retrieval = ninfer::models::qwen3_5::detail::KVRetrieval::instance();
        retrieval != nullptr) {
        retrieval->accumulate_keys(kn.data, fidx, s);
        retrieval->accumulate_query(qn.data, fidx, s);
    }
    debug_layer_checksum("k-rope", fidx, kn, s);
    if (std::getenv("NINFER_ATTN_PROBE") != nullptr && !s_rope_pre_qn.empty()) {
        // The post-rope side: fire only when we actually captured a pre copy. Write both
        // files together and clear the buffer so a later call can't pair stale data.
        const char* pf = (ph == Phase::Prefill) ? "pre" : "dec";
        const std::size_t qn2 = static_cast<std::size_t>(qn.ne[0]) * qn.ne[1] * qn.ne[2];
        std::vector<std::uint16_t> qv2(qn2);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(qv2.data(), qn.data, qn2 * 2, cudaMemcpyDeviceToHost));
        std::string preq = "D:/ninfer-hip-migrate/build/attn_qn_norope_"; preq += pf;
        preq += ".bin";
        std::FILE* pf2 = std::fopen(preq.c_str(), "wb");
        if (pf2) { std::fwrite(s_rope_pre_qn.data(), 2, s_rope_pre_qn.size(), pf2);
                   std::fclose(pf2); }
        std::string postq = "D:/ninfer-hip-migrate/build/attn_qn_rope_"; postq += pf;
        postq += ".bin";
        std::FILE* qf2 = std::fopen(postq.c_str(), "wb");
        if (qf2) { std::fwrite(qv2.data(), 2, qn2, qf2); std::fclose(qf2); }
        const std::size_t kn2 = static_cast<std::size_t>(kn.ne[0]) * kn.ne[1] * kn.ne[2];
        std::vector<std::uint16_t> kv2(kn2);
        CUDA_CHECK(cudaMemcpy(kv2.data(), kn.data, kn2 * 2, cudaMemcpyDeviceToHost));
        std::string prek = "D:/ninfer-hip-migrate/build/attn_kn_norope_"; prek += pf;
        prek += ".bin";
        std::FILE* kf0 = std::fopen(prek.c_str(), "wb");
        if (kf0) { std::fwrite(s_rope_pre_kn.data(), 2, s_rope_pre_kn.size(), kf0);
                   std::fclose(kf0); }
        std::string postk = "D:/ninfer-hip-migrate/build/attn_kn_rope_"; postk += pf;
        postk += ".bin";
        std::FILE* kf2 = std::fopen(postk.c_str(), "wb");
        if (kf2) { std::fwrite(kv2.data(), 2, kn2, kf2); std::fclose(kf2); }
        // report pre/post norms on the same call (norm preservation check)
        double npre = 0.0, npost = 0.0;
        for (std::size_t i = 0; i < s_rope_pre_qn.size(); ++i) {
            const float a = (static_cast<float>(s_rope_pre_qn[i]) *
                             (1.0f / 32768.0f));
            const float b = (static_cast<float>(qv2[i]) * (1.0f / 32768.0f));
            npre += static_cast<double>(a) * a;
            npost += static_cast<double>(b) * b;
        }
        std::fprintf(stderr, "[ropeprobe] %s post-fire qn2=%zu npre=%.4f npost=%.4f ratio=%.6f\n",
                     pf, qn2, std::sqrt(npre), std::sqrt(npost), std::sqrt(npost / (npre + 1e-30)));
        std::fflush(stderr);
        s_rope_pre_qn.clear();
        s_rope_pre_kn.clear();
    }

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    // TEMP: NINFER_ATTN_FAKE=1 skips the attention kernel to measure the projection +
    // o_proj share of the attn mixer (timing only; output is wrong).
    static const bool fake_attn = [] {
        const char* e = std::getenv("NINFER_ATTN_FAKE");
        return e != nullptr && e[0] == '1';
    }();
    if (fake_attn) {
        const std::size_t a_elems = static_cast<std::size_t>(a.ne[0]) * a.ne[1] * a.ne[2] * a.ne[3];
        cudaMemsetAsync(a.data, 0, a_elems * 2, s);
    } else if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = cache_positions_ptr->view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_,
            a_batch, s);
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, *cache_positions_ptr, Tensor{}, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a,
            s);
    }
    // dump raw attention output (before gate) + inputs for offline CPU verification
    if (std::getenv("NINFER_ATTN_PROBE") != nullptr) {
        static int attn_pre_dumped = 0;
        static int attn_dec_step = 0;
        const bool take_post = (ph == Phase::Prefill && attn_pre_dumped == 0) ||
                               (ph == Phase::Verify && fidx == 0);
        if (take_post) {
            if (ph == Phase::Prefill) { ++attn_pre_dumped; }
            const char* pf = ph == Phase::Prefill ? "pre" : "dec";
            std::string step_sfx;
            if (ph == Phase::Verify) {
                step_sfx = "_s" + std::to_string(attn_dec_step);
                if (fidx == 0) ++attn_dec_step;
            }
            auto n3 = [](const Tensor& t) {
                return static_cast<std::size_t>(t.ne[0]) * t.ne[1] * t.ne[2];
            };
            const std::size_t an = n3(a);
            std::vector<std::uint16_t> av(an);
            const std::size_t qn2 = n3(qn);
            std::vector<std::uint16_t> qv(qn2);
            const std::size_t kn2 = n3(kn);
            std::vector<std::uint16_t> kv2(kn2);
            const std::size_t vn = n3(v);
            std::vector<std::uint16_t> vv2(vn);
            const std::size_t cn = static_cast<std::size_t>(cache_positions.ne[0]) *
                                    cache_positions.ne[1];
            std::vector<std::int32_t> cpos(cn);
            std::vector<std::int32_t> rpos(cn);
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaMemcpy(av.data(), a.data, an * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(qv.data(), qn.data, qn2 * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(kv2.data(), kn.data, kn2 * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vv2.data(), v.data, vn * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(cpos.data(), cache_positions.data, cn * 4,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(rpos.data(), rope_for_op.data, cn * 4,
                                  cudaMemcpyDeviceToHost));
            std::string ap = "D:/ninfer-hip-migrate/build/attn_a_raw_"; ap += pf; ap += ".bin";
            if (ph == Phase::Verify) { ap = "D:/ninfer-hip-migrate/build/attn_a_raw_dec" + step_sfx + ".bin"; }
            std::FILE* af = std::fopen(ap.c_str(), "wb");
            if (af) { std::fwrite(av.data(), 2, an, af); std::fclose(af); }
            std::string qp = "D:/ninfer-hip-migrate/build/attn_qn_"; qp += pf; qp += ".bin";
            if (ph == Phase::Verify) { qp = "D:/ninfer-hip-migrate/build/attn_qn_dec" + step_sfx + ".bin"; }
            std::FILE* qf = std::fopen(qp.c_str(), "wb");
            if (qf) { std::fwrite(qv.data(), 2, qn2, qf); std::fclose(qf); }
            std::string kp = "D:/ninfer-hip-migrate/build/attn_kn_"; kp += pf; kp += ".bin";
            if (ph == Phase::Verify) { kp = "D:/ninfer-hip-migrate/build/attn_kn_dec" + step_sfx + ".bin"; }
            std::FILE* kf = std::fopen(kp.c_str(), "wb");
            if (kf) { std::fwrite(kv2.data(), 2, kn2, kf); std::fclose(kf); }
            std::string vp = "D:/ninfer-hip-migrate/build/attn_v_"; vp += pf; vp += ".bin";
            if (ph == Phase::Verify) { vp = "D:/ninfer-hip-migrate/build/attn_v_dec" + step_sfx + ".bin"; }
            std::FILE* vf = std::fopen(vp.c_str(), "wb");
            if (vf) { std::fwrite(vv2.data(), 2, vn, vf); std::fclose(vf); }
            std::string cp = "D:/ninfer-hip-migrate/build/attn_pos_"; cp += pf; cp += ".bin";
            if (ph == Phase::Verify) { cp = "D:/ninfer-hip-migrate/build/attn_pos_dec" + step_sfx + ".bin"; }
            std::FILE* cf = std::fopen(cp.c_str(), "wb");
            if (cf) { std::fwrite(cpos.data(), 4, cn, cf); std::fclose(cf); }
            std::string rp = "D:/ninfer-hip-migrate/build/attn_rope_"; rp += pf; rp += ".bin";
            if (ph == Phase::Verify) { rp = "D:/ninfer-hip-migrate/build/attn_rope_dec" + step_sfx + ".bin"; }
            std::FILE* rf = std::fopen(rp.c_str(), "wb");
            if (rf) { std::fwrite(rpos.data(), 4, cn, rf); std::fclose(rf); }
            std::fprintf(stderr,
                         "[attnprobe] %s fi=%d T=%d a ne=[%d,%d] qn=[%d,%d] kn=[%d,%d] v=[%d,%d] "
                         "cachepos=%zu rope=%zu\n",
                         ph == Phase::Prefill ? "PREFILL" : "DECODE", fidx, T, a.ne[0], a.ne[1],
                         qn.ne[0], qn.ne[1], kn.ne[0], kn.ne[1], v.ne[0], v.ne[1], cn, cn);
            std::fflush(stderr);
        }
    }
    debug_layer_checksum("attn-raw", 0, a, s);
    debug_layer_checksum("gate", 0, gate, s);
    ops::sigmoid_mul(gate, a, s);
    debug_layer_checksum("attn-out", 0, a, s);

    ops::linear_add(a.view({dimension(config_.attention->query_width()), T}), p.output.weight, x,
                    p.output.policy, work_, s);
    debug_layer_checksum("x-add", 0, x, s);
    if (std::getenv("NINFER_ATTN_XADD_DUMP") != nullptr && ph == Phase::Verify) {
        // Dump every attention layer's x_add (decode) so the CPU replay can chain
        // across the attn layer into the next GDN block. Filename has fidx so no
        // step collisions.
        const std::size_t xn =
            static_cast<std::size_t>(x.ne[0]) * x.ne[1] * (x.ne[2] > 0 ? x.ne[2] : 1);
        std::vector<std::uint16_t> xv(xn);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(xv.data(), x.data, xn * 2, cudaMemcpyDeviceToHost));
        std::string xp = "D:/ninfer-hip-migrate/build/attn_xadd_dec_f" + std::to_string(fidx) +
                         ".bin";
        std::FILE* xf2 = std::fopen(xp.c_str(), "wb");
        if (xf2) { std::fwrite(xv.data(), 2, xn, xf2); std::fclose(xf2); }
        std::fprintf(stderr, "[attnxadd] fi=%d dumped %zu\n", fidx, xn);
        std::fflush(stderr);
    }
}

void TextContext::gdn_mix(const BlockParameters& w, Tensor& x, int gidx, Phase ph) {
    const auto& p  = std::get<GdnParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    debug_layer_checksum("gdn-in", gidx, x, s);
    // Decode step counter: each decode step's first GDN layer (gidx==0) bumps the
    // step; the other 47 layers of the same step reuse it. (Position-based counting
    // was unreliable because active_cache_positions_ / io_.pos are not stable in
    // the verify phase.)
    static std::int32_t gdn_decode_step = 0;
    if (ph == Phase::Verify && gidx == 0) { ++gdn_decode_step; }

    const auto control = workspace::gdn_control(work_, config_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    if (std::getenv("NINFER_GDN_XPROBE") != nullptr && gidx == 0) {
        cudaStreamSynchronize(s);
        std::fprintf(stderr, "[gdn-xprobe] gidx=%d x ne=[%d,%d] nb1=%lld data=%p numel=%zu\n",
                     gidx, x.ne[0], x.ne[1],
                     static_cast<long long>(x.nb[1]), static_cast<void*>(x.data),
                     static_cast<std::size_t>(x.numel()));
        std::vector<std::uint16_t> xv(static_cast<std::size_t>(x.ne[0]) * x.ne[1]);
        CUDA_CHECK(cudaMemcpy(xv.data(), x.data, xv.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* xf = std::fopen("D:/ninfer-hip-migrate/build/gdn_xpre_0.bin", "wb");
        if (xf) { std::fwrite(xv.data(), 2, xv.size(), xf); std::fclose(xf); }
        std::fprintf(stderr, "[gdn-xprobe] gidx=%d x ne=[%d,%d] dumped\n", gidx, x.ne[0], x.ne[1]);
        std::fflush(stderr);
    }
    static const bool skip_gdnctl = std::getenv("NINFER_SKIP_GDNCTL") != nullptr;
    if (!skip_gdnctl) {
        gdn_norm_control(x, w.input_norm, config_.rms_norm_eps, p, h, g, beta, work_,
                         ctx_.execution_view());
    }
    if (std::getenv("NINFER_CONV_PROBE") != nullptr && gidx == 0) {
        const cudaError_t sync = cudaStreamSynchronize(s);
        std::fprintf(stderr, "[convprobe] post-norm_control sync=%d (%s)\n",
                     static_cast<int>(sync), cudaGetErrorString(sync));
        std::vector<float> gv(static_cast<std::size_t>(g.ne[0]) * g.ne[1]);
        std::vector<float> bv(static_cast<std::size_t>(beta.ne[0]) * beta.ne[1]);
        CUDA_CHECK(cudaMemcpy(gv.data(), g.data, gv.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bv.data(), beta.data, bv.size() * 4, cudaMemcpyDeviceToHost));
        std::FILE* gf = std::fopen(ph == Phase::Prefill ? "D:/ninfer-hip-migrate/build/gdn_g_pf.bin"
                                                         : "D:/ninfer-hip-migrate/build/gdn_g_dec.bin",
                                   "wb");
        if (gf) { std::fwrite(gv.data(), 4, gv.size(), gf); std::fclose(gf); }
        std::FILE* bf = std::fopen(ph == Phase::Prefill ? "D:/ninfer-hip-migrate/build/gdn_beta_pf.bin"
                                                        : "D:/ninfer-hip-migrate/build/gdn_beta_dec.bin",
                                   "wb");
        if (bf) { std::fwrite(bv.data(), 4, bv.size(), bf); std::fclose(bf); }
        std::vector<std::uint16_t> hv(static_cast<std::size_t>(h.ne[0]) * h.ne[1]);
        CUDA_CHECK(cudaMemcpy(hv.data(), h.data, hv.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* hf = std::fopen("D:/ninfer-hip-migrate/build/gdn_h.bin", "wb");
        if (hf) { std::fwrite(hv.data(), 2, hv.size(), hf); std::fclose(hf); }
        std::fprintf(stderr, "[convprobe] g ne=[%d,%d] beta ne=[%d,%d] h ne=[%d,%d] dumped\n",
                     g.ne[0], g.ne[1], beta.ne[0], beta.ne[1], h.ne[0], h.ne[1]);
        std::fflush(stderr);
    }

    const auto projection = workspace::gdn_projection(work_, config_, T);
    Tensor z  = projection.output_gate.view({dimension(config_.gdn->linear_value_head_dim),
                                             dimension(config_.gdn->linear_num_value_heads), T});
    Tensor qc = projection.query;
    Tensor kc = projection.key;
    Tensor vc = projection.value;
    if (ph == Phase::Prefill && std::getenv("NINFER_CONV_PROBE") != nullptr) {
        cudaStreamSynchronize(s);
        auto dump_u16 = [&](const Tensor& t, const char* fn) {
            std::vector<std::uint16_t> buf(static_cast<std::size_t>(t.numel()));
            CUDA_CHECK(cudaMemcpy(buf.data(), t.data, buf.size() * 2, cudaMemcpyDeviceToHost));
            std::FILE* f = std::fopen(fn, "wb");
            if (f) { std::fwrite(buf.data(), 2, buf.size(), f); std::fclose(f); }
        };
        char qfn[96], kfn[96], vfn[96], zfn[96], hfn[96];
        std::snprintf(qfn, sizeof(qfn), "D:/ninfer-hip-migrate/build/gdn_pf_q_%d_early.bin", gidx);
        std::snprintf(kfn, sizeof(kfn), "D:/ninfer-hip-migrate/build/gdn_pf_k_%d_early.bin", gidx);
        std::snprintf(vfn, sizeof(vfn), "D:/ninfer-hip-migrate/build/gdn_pf_v_%d_early.bin", gidx);
        std::snprintf(zfn, sizeof(zfn), "D:/ninfer-hip-migrate/build/gdn_pf_z_%d_early.bin", gidx);
        std::snprintf(hfn, sizeof(hfn), "D:/ninfer-hip-migrate/build/gdn_pf_h_%d_early.bin", gidx);
        dump_u16(qc, qfn); dump_u16(kc, kfn); dump_u16(vc, vfn); dump_u16(z, zfn); dump_u16(h, hfn);
        std::fprintf(stderr, "[convprobe-pf] gidx=%d T=%d q/k/v/z/h dumped\n", gidx, T);
        std::fflush(stderr);
    }
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor projection_input =
            h.view({dimension(config_.hidden_size), width, active_sequence_batch_});
        Tensor query_output =
            qc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor key_output =
            kc.view({dimension(config_.gdn->key_width()), width, active_sequence_batch_});
        Tensor value_output =
            vc.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor gate_output =
            z.view({dimension(config_.gdn->value_width()), width, active_sequence_batch_});
        Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        if (std::getenv("NINFER_CONV_STATE_DUMP") != nullptr) {
            cudaStreamSynchronize(s);
            std::int32_t cslot = 0;
            std::int32_t dslot = 0;
            CUDA_CHECK(cudaMemcpy(&cslot, active_linear_state_source_slots_->data, sizeof(cslot),
                                  cudaMemcpyDeviceToHost));
            if (active_linear_state_destination_slots_ != nullptr) {
                CUDA_CHECK(cudaMemcpy(&dslot,
                                      active_linear_state_destination_slots_->data, sizeof(dslot),
                                      cudaMemcpyDeviceToHost));
            }
            Tensor source_conv = state_.conv_slot(static_cast<std::uint32_t>(gidx), cslot);
            Tensor dest_conv   = state_.conv_slot(static_cast<std::uint32_t>(gidx), dslot);
            const std::size_t cn = static_cast<std::size_t>(source_conv.numel());
            std::vector<std::uint16_t> cv(cn);
            std::vector<std::uint16_t> dv(cn);
            CUDA_CHECK(cudaMemcpy(cv.data(), source_conv.data, cn * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dv.data(), dest_conv.data, cn * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            std::size_t nz = 0;
            float lo = 1.0e30f, hi = -1.0e30f;
            for (const auto bits : cv) {
                if (bits != 0) ++nz;
                const float value = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
                lo = std::min(lo, value); hi = std::max(hi, value);
            }
            char csf[128];
            std::snprintf(csf, sizeof(csf), "D:/ninfer-hip-migrate/build/conv_state_in_%d_s%d.bin",
                          gidx, gdn_decode_step);
            std::FILE* cvf = std::fopen(csf, "wb");
            if (cvf) { std::fwrite(cv.data(), sizeof(std::uint16_t), cn, cvf); std::fclose(cvf); }
            std::FILE* dsf = std::fopen("D:/ninfer-hip-migrate/build/conv_state_out_s%d.bin",
                                        "wb");
            if (dsf) { std::fwrite(dv.data(), sizeof(std::uint16_t), cn, dsf); std::fclose(dsf); }
            std::fprintf(stderr, "[convprobe-verify] gidx=%d step=%d src_slot=%d dst_slot=%d conv=[%d,%d] "
                                 "in_nz=%zu/%zu range=[%g,%g]\n",
                         gidx, gdn_decode_step, cslot, dslot, source_conv.ne[0],
                         source_conv.ne[1], nz, cn, lo, hi);
            std::fflush(stderr);
        }
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            gdn_projection_record(projection_input, p, *config_.gdn, conv_states, valid,
                                  *active_linear_state_source_slots_, records.conv, query_output,
                                  key_output, value_output, gate_output, work_, s);
        } else {
            // Ablation hook (NINFER_SKIP_GDNPROJ=1): drop the fused qkv/z projection + conv
            // so the remaining GDN cost can be attributed. Timing only.
            static const bool skip_gdnproj = std::getenv("NINFER_SKIP_GDNPROJ") != nullptr;
            if (!skip_gdnproj) {
                gdn_projection_snapshot(projection_input, p, *config_.gdn, conv_states, valid,
                                        *active_linear_state_source_slots_,
                                        *active_linear_state_destination_slots_, query_output,
                                        key_output, value_output, gate_output, work_, s);
            }
        }
        if (std::getenv("NINFER_CONV_PROBE") != nullptr) {
            cudaStreamSynchronize(s);
            auto dump_u16 = [&](const Tensor& t, const char* fn) {
                std::vector<std::uint16_t> buf(static_cast<std::size_t>(t.numel()));
                CUDA_CHECK(cudaMemcpy(buf.data(), t.data, buf.size() * 2, cudaMemcpyDeviceToHost));
                std::FILE* f = std::fopen(fn, "wb");
                if (f) { std::fwrite(buf.data(), 2, buf.size(), f); std::fclose(f); }
                return buf;
            };
            char qfn[96], kfn[96], vfn[96], zfn[96], hfn[96];
            std::snprintf(qfn, sizeof(qfn), "D:/ninfer-hip-migrate/build/gdn_q_%d_s%d.bin", gidx, gdn_decode_step);
            std::snprintf(kfn, sizeof(kfn), "D:/ninfer-hip-migrate/build/gdn_k_%d_s%d.bin", gidx, gdn_decode_step);
            std::snprintf(vfn, sizeof(vfn), "D:/ninfer-hip-migrate/build/gdn_v_%d_s%d.bin", gidx, gdn_decode_step);
            std::snprintf(zfn, sizeof(zfn), "D:/ninfer-hip-migrate/build/gdn_z_%d_s%d.bin", gidx, gdn_decode_step);
            std::snprintf(hfn, sizeof(hfn), "D:/ninfer-hip-migrate/build/gdn_h_%d_s%d.bin", gidx, gdn_decode_step);
            dump_u16(query_output, qfn);
            dump_u16(key_output, kfn);
            dump_u16(value_output, vfn);
            dump_u16(gate_output, zfn);
            dump_u16(h, hfn);
            std::vector<float> gv(static_cast<std::size_t>(g.numel())),
                bv(static_cast<std::size_t>(beta.numel()));
            CUDA_CHECK(cudaMemcpy(gv.data(), g.data, gv.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(bv.data(), beta.data, bv.size() * 4, cudaMemcpyDeviceToHost));
            char gfn[96], bfn[96];
            std::snprintf(gfn, sizeof(gfn), "D:/ninfer-hip-migrate/build/gdn_g_%d_s%d.bin", gidx, gdn_decode_step);
            std::snprintf(bfn, sizeof(bfn), "D:/ninfer-hip-migrate/build/gdn_beta_%d_s%d.bin", gidx, gdn_decode_step);
            std::FILE* f1 = std::fopen(gfn, "wb");
            if (f1) { std::fwrite(gv.data(), 4, gv.size(), f1); std::fclose(f1); }
            std::FILE* f2 = std::fopen(bfn, "wb");
            if (f2) { std::fwrite(bv.data(), 4, bv.size(), f2); std::fclose(f2); }
            std::fprintf(stderr, "[convprobe-verify-post] gidx=%d q=%zu k=%zu v=%zu z=%zu h=%zu g=%zu\n",
                         gidx,
                         static_cast<std::size_t>(query_output.numel()),
                         static_cast<std::size_t>(key_output.numel()),
                         static_cast<std::size_t>(value_output.numel()),
                         static_cast<std::size_t>(gate_output.numel()),
                         static_cast<std::size_t>(h.numel()), gv.size());
            std::fflush(stderr);
        }
    } else {
        Tensor qkv    = workspace::gdn_prefill_conv(work_, config_, T);
        Tensor z_flat = z.view({dimension(config_.gdn->value_width()), T});
        gdn_projection(h, p, qkv, z_flat, work_, s);
        if (std::getenv("NINFER_CONV_PROBE") != nullptr && gidx == 0) {
            // The pre-conv CUDA_CHECK false-negative: hipGetLastError never reflects a stream
            // failure; only the Synchronize return value does. A failed stream poisons every
            // later launch on it with hipErrorLaunchFailure.
            const cudaError_t sync = cudaStreamSynchronize(s);
            const cudaError_t e0   = cudaGetLastError();
            std::fprintf(stderr, "[convprobe] pre-conv sync=%d (%s) lastError=%d (%s)\n",
                         static_cast<int>(sync), cudaGetErrorString(sync), static_cast<int>(e0),
                         cudaGetErrorString(e0));
            std::fflush(stderr);
        }
                Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        if (std::getenv("NINFER_CONV_STATE_DUMP") != nullptr && gidx == 0) {
            std::fprintf(stderr, "[convprobe] qkv=%p ne[0]=%d ne[1]=%d | qc=%p kc=%p vc=%p | "
                                 "conv_w=%p state_in=%p state_out=%p conv_state ne=[%d,%d]\n",
                         static_cast<void*>(qkv.data), qkv.ne[0], qkv.ne[1],
                         static_cast<void*>(qc.data), static_cast<void*>(kc.data),
                         static_cast<void*>(vc.data), static_cast<void*>(p.convolution.data),
                         static_cast<void*>(conv_state_in.data),
                         static_cast<void*>(conv_state_out.data), conv_state_in.ne[0],
                         conv_state_in.ne[1]);
            std::vector<std::uint16_t> st(static_cast<std::size_t>(conv_state_in.ne[0]) *
                                          conv_state_in.ne[1]);
            CUDA_CHECK(cudaMemcpy(st.data(), conv_state_in.data, st.size() * 2,
                                  cudaMemcpyDeviceToHost));
            std::FILE* stf = std::fopen(ph == Phase::Prefill
                                            ? "D:/ninfer-hip-migrate/build/conv_state_in_pf.bin"
                                            : "D:/ninfer-hip-migrate/build/conv_state_in_dec.bin",
                                        "wb");
            if (stf) {
                std::fwrite(st.data(), 2, st.size(), stf);
                std::fclose(stf);
            }
            std::size_t nonzero = 0;
            for (const auto v : st) {
                if (v != 0) { ++nonzero; }
            }
            std::fprintf(stderr, "[convprobe] state_in dumped %zu elems nonzero=%zu (%.1f%%)\n",
                         st.size(), nonzero, st.empty() ? 0.0 : 100.0 * double(nonzero) / st.size());
            std::fflush(stderr);
        }
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc, kc, vc,
                                      s);
        if (std::getenv("NINFER_CONV_STATE_DUMP") != nullptr && gidx == 0) {
            cudaStreamSynchronize(s);
            std::fprintf(stderr, "[convprobe-post] source_slot=%d dest_slot=%d conv_state_out ne=[%d,%d] "
                                 "state_in.data=%p state_out.data=%p\n",
                         linear_state_source_slot_, linear_state_destination_slot_,
                         conv_state_out.ne[0], conv_state_out.ne[1],
                         static_cast<void*>(conv_state_in.data),
                         static_cast<void*>(conv_state_out.data));
            const std::size_t on = static_cast<std::size_t>(conv_state_out.ne[0]) *
                                   conv_state_out.ne[1];
            std::vector<std::uint16_t> so(on);
            CUDA_CHECK(cudaMemcpy(so.data(), conv_state_out.data, so.size() * 2,
                                  cudaMemcpyDeviceToHost));
            std::FILE* sof = std::fopen(ph == Phase::Prefill
                                             ? "D:/ninfer-hip-migrate/build/conv_state_out_pf.bin"
                                             : "D:/ninfer-hip-migrate/build/conv_state_out_dec.bin",
                                         "wb");
            if (sof) { std::fwrite(so.data(), 2, so.size(), sof); std::fclose(sof); }
            std::size_t nz0 = 0;
            for (std::size_t i = 0; i < on; ++i) { if (so[i] != 0) ++nz0; }
            std::fprintf(stderr, "[convprobe-post] conv_state_out dumped %zu elems nonzero=%zu (%.1f%%)\n",
                         on, nz0, on == 0 ? 0.0 : 100.0 * double(nz0) / double(on));
            std::fflush(stderr);
        }
        if (std::getenv("NINFER_CONV_PROBE") != nullptr && gidx == 0) {
            cudaStreamSynchronize(s);
        std::vector<std::uint16_t> hv(static_cast<std::size_t>(h.ne[0]) * h.ne[1]);
        CUDA_CHECK(cudaMemcpy(hv.data(), h.data, hv.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* hf = std::fopen("D:/ninfer-hip-migrate/build/gdn_h.bin", "wb");
        if (hf) { std::fwrite(hv.data(), 2, hv.size(), hf); std::fclose(hf); }
        std::vector<std::uint16_t> qkv_dump(static_cast<std::size_t>(qkv.ne[0]) * qkv.ne[1]);
        CUDA_CHECK(cudaMemcpy(qkv_dump.data(), qkv.data, qkv_dump.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* qkvc = std::fopen("D:/ninfer-hip-migrate/build/gdn_qkv.bin", "wb");
        if (qkvc) { std::fwrite(qkv_dump.data(), 2, qkv_dump.size(), qkvc); std::fclose(qkvc); }
        std::vector<std::uint16_t> inw_dump(static_cast<std::size_t>(w.input_norm.ne[0]));
        CUDA_CHECK(cudaMemcpy(inw_dump.data(), w.input_norm.data, inw_dump.size() * 2,
                              cudaMemcpyDeviceToHost));
        std::FILE* inwf = std::fopen("D:/ninfer-hip-migrate/build/gdn_inw.bin", "wb");
        if (inwf) { std::fwrite(inw_dump.data(), 2, inw_dump.size(), inwf); std::fclose(inwf); }
        std::vector<std::uint16_t> q_dump(static_cast<std::size_t>(qc.ne[0]) * qc.ne[1]);
        CUDA_CHECK(cudaMemcpy(q_dump.data(), qc.data, q_dump.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* qf = std::fopen(ph == Phase::Prefill
                                       ? "D:/ninfer-hip-migrate/build/gdn_q_pf.bin"
                                       : "D:/ninfer-hip-migrate/build/gdn_q_dec.bin",
                                   "wb");
        if (qf) { std::fwrite(q_dump.data(), 2, q_dump.size(), qf); std::fclose(qf); }
        std::vector<std::uint16_t> k_dump(static_cast<std::size_t>(kc.ne[0]) * kc.ne[1]);
        CUDA_CHECK(cudaMemcpy(k_dump.data(), kc.data, k_dump.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* kf = std::fopen(ph == Phase::Prefill
                                       ? "D:/ninfer-hip-migrate/build/gdn_k_pf.bin"
                                       : "D:/ninfer-hip-migrate/build/gdn_k_dec.bin",
                                   "wb");
        if (kf) { std::fwrite(k_dump.data(), 2, k_dump.size(), kf); std::fclose(kf); }
        std::vector<std::uint16_t> v_dump(static_cast<std::size_t>(vc.ne[0]) * vc.ne[1]);
        CUDA_CHECK(cudaMemcpy(v_dump.data(), vc.data, v_dump.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* vf = std::fopen(ph == Phase::Prefill
                                       ? "D:/ninfer-hip-migrate/build/gdn_v_pf.bin"
                                       : "D:/ninfer-hip-migrate/build/gdn_v_dec.bin",
                                   "wb");
        if (vf) { std::fwrite(v_dump.data(), 2, v_dump.size(), vf); std::fclose(vf); }
        }
        if (std::getenv("NINFER_CONV_PROBE") != nullptr && gidx > 0) {
            cudaStreamSynchronize(s);
            auto dump_u16_p = [&](const Tensor& t, const char* fn) {
                std::vector<std::uint16_t> buf(static_cast<std::size_t>(t.numel()));
                CUDA_CHECK(cudaMemcpy(buf.data(), t.data, buf.size() * 2, cudaMemcpyDeviceToHost));
                std::FILE* f = std::fopen(fn, "wb");
                if (f) { std::fwrite(buf.data(), 2, buf.size(), f); std::fclose(f); }
            };
            char qfn2[96], kfn2[96], vfn2[96], hfn2[96];
            std::snprintf(qfn2, sizeof(qfn2), "D:/ninfer-hip-migrate/build/gdn_pf_q_%d.bin", gidx);
            std::snprintf(kfn2, sizeof(kfn2), "D:/ninfer-hip-migrate/build/gdn_pf_k_%d.bin", gidx);
            std::snprintf(vfn2, sizeof(vfn2), "D:/ninfer-hip-migrate/build/gdn_pf_v_%d.bin", gidx);
            std::snprintf(hfn2, sizeof(hfn2), "D:/ninfer-hip-migrate/build/gdn_pf_h_%d.bin", gidx);
            dump_u16_p(qc, qfn2); dump_u16_p(kc, kfn2); dump_u16_p(vc, vfn2); dump_u16_p(h, hfn2);
            std::fprintf(stderr, "[convprobe-pf] gidx=%d T=%d q/k/v/h dumped\n", gidx, T);
            std::fflush(stderr);
        }
    }

    Tensor q_recurrent = qc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});
    Tensor k_recurrent = kc.view({dimension(config_.gdn->linear_key_head_dim),
                                  dimension(config_.gdn->linear_num_key_heads), T});

    Tensor vv = vc.view({dimension(config_.gdn->linear_value_head_dim),
                         dimension(config_.gdn->linear_num_value_heads), T});
    Tensor o  = workspace::gdn_recurrent_output(work_, config_, T)
                   .view({dimension(config_.gdn->linear_value_head_dim),
                          dimension(config_.gdn->linear_num_value_heads), T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        if (std::getenv("NINFER_GDN_STATE_PROBE") != nullptr) {
            // decode: dump every gdn layer's recurrent state source slot (all 48
            // layers) per decode step, so the CPU reference can compare each layer's
            // state against the previous step's updated state.
            cudaStreamSynchronize(s);
            std::int32_t slot = 0;
            CUDA_CHECK(cudaMemcpy(&slot, active_linear_state_source_slots_->data, sizeof(slot),
                                  cudaMemcpyDeviceToHost));
            Tensor source_state =
                state_.recurrent_slot(static_cast<std::uint32_t>(gidx), slot);
            const std::size_t rec_n = static_cast<std::size_t>(source_state.numel());
            std::vector<float> rec_buf(rec_n);
            CUDA_CHECK(cudaMemcpy(rec_buf.data(), source_state.data, rec_n * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            std::size_t nz = 0, nf = 0;
            for (const auto v : rec_buf) { if (v != 0.0f) ++nz; if (!std::isfinite(v)) ++nf; }
            char sf[128];
            std::snprintf(sf, sizeof(sf), "D:/ninfer-hip-migrate/build/gdn_state_in_%d_s%d.bin",
                          gidx, gdn_decode_step);
            std::FILE* recf = std::fopen(sf, "wb");
            if (recf) { std::fwrite(rec_buf.data(), sizeof(float), rec_n, recf); std::fclose(recf); }
            std::fprintf(stderr, "[stateprobe-verify] gidx=%d step=%d slot=%d dtype=%d ne=[%d,%d,%d] "
                                 "nonzero=%zu/%zu nonfinite=%zu range=[%g,%g]\n",
                         gidx, gdn_decode_step, slot, static_cast<int>(source_state.dtype),
                         recurrent_states.ne[0], recurrent_states.ne[1], recurrent_states.ne[2],
                         nz, rec_n, nf, rec_n ? *std::min_element(rec_buf.begin(), rec_buf.end()) : 0.0f,
                         rec_n ? *std::max_element(rec_buf.begin(), rec_buf.end()) : 0.0f);
            std::fflush(stderr);
        }
        Tensor q_batch           = q_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor k_batch           = k_recurrent.view({dimension(config_.gdn->linear_key_head_dim),
                                                     dimension(config_.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor v_batch           = vv.view({dimension(config_.gdn->linear_value_head_dim),
                                            dimension(config_.gdn->linear_num_value_heads), width,
                                            active_sequence_batch_});
        Tensor g_batch =
            g.view({dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor beta_batch = beta.view(
            {dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor out_batch =
            o.view({dimension(config_.gdn->linear_value_head_dim),
                    dimension(config_.gdn->linear_num_value_heads), width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                recurrent_states, valid, *active_linear_state_source_slots_, records.key,
                records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
        if (std::getenv("NINFER_GDN_STATE_PROBE") != nullptr) {
            cudaStreamSynchronize(s);
            const std::size_t on2 = static_cast<std::size_t>(out_batch.numel());
            std::vector<std::uint16_t> ob(on2);
            CUDA_CHECK(cudaMemcpy(ob.data(), out_batch.data, on2 * 2, cudaMemcpyDeviceToHost));
            char ofn[96];
            std::snprintf(ofn, sizeof(ofn), "D:/ninfer-hip-migrate/build/gdn_o_%d_s%d.bin",
                          gidx, gdn_decode_step);
            std::FILE* of = std::fopen(ofn, "wb");
            if (of) { std::fwrite(ob.data(), 2, on2, of); std::fclose(of); }
            std::size_t nz = 0;
            for (const auto v : ob) { if (v != 0) ++nz; }
            std::fprintf(stderr, "[oprobe-verify] gidx=%d out_batch=%zu nonzero=%zu range0=%u,%u,%u,%u\n",
                         gidx, on2, nz, static_cast<unsigned>(ob[0]), static_cast<unsigned>(ob[1]),
                         static_cast<unsigned>(ob[2]), static_cast<unsigned>(ob[3]));
            std::fflush(stderr);
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        if (std::getenv("NINFER_GDN_STATE_PROBE") != nullptr && gidx == 0) {
            cudaStreamSynchronize(s);
            const std::size_t rec_n = static_cast<std::size_t>(recurrent_state_in.ne[0]) *
                                      recurrent_state_in.ne[1] * recurrent_state_in.ne[2];
            std::vector<std::uint16_t> rec_buf(rec_n);
            CUDA_CHECK(cudaMemcpy(rec_buf.data(), recurrent_state_in.data, rec_n * 2,
                                  cudaMemcpyDeviceToHost));
            std::FILE* recf = std::fopen("D:/ninfer-hip-migrate/build/gdn_state_in.bin", "wb");
            if (recf) { std::fwrite(rec_buf.data(), 2, rec_n, recf); std::fclose(recf); }
            std::size_t nz = 0;
            for (std::size_t i = 0; i < rec_n; ++i) { if (rec_buf[i] != 0) ++nz; }
            std::fprintf(stderr, "[stateprobe] gidx=%d recurrent_state_in ne=[%d,%d,%d] "
                                 "nonzero=%zu/%zu (%.1f%%)\n",
                         gidx, recurrent_state_in.ne[0], recurrent_state_in.ne[1],
                         recurrent_state_in.ne[2], nz, rec_n,
                         rec_n == 0 ? 0.0 : 100.0 * double(nz) / double(rec_n));
            std::fflush(stderr);
        }
        // Ablation hook (NINFER_SKIP_GDN=1): drop the GDN recurrence so its share of the
        // decode step can be measured directly. Output is garbage; timing is the point.
        static const bool skip_gdn = std::getenv("NINFER_SKIP_GDN") != nullptr;
        if (!skip_gdn) {
            ops::gated_delta_net(
                q_recurrent, k_recurrent, vv, g, beta,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config_.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out, o, s);
        }
    }

    Tensor on = workspace::gdn_normalized_output(work_, config_, T)
                    .view({dimension(config_.gdn->linear_value_head_dim),
                           dimension(config_.gdn->linear_num_value_heads), T});
    static const bool skip_grms = std::getenv("NINFER_SKIP_GRMS") != nullptr;
    if (!skip_grms) {
        ops::gated_rmsnorm(o, p.norm, z, config_.rms_norm_eps, on, s);
    }

    if (std::getenv("NINFER_GDN_STATE_PROBE") != nullptr) {
        // per-layer gated_rmsnorm output on (post-norm, pre-o_proj) for isolation.
        cudaStreamSynchronize(s);
        const std::size_t onn = static_cast<std::size_t>(on.numel());
        std::vector<std::uint16_t> onb(onn);
        CUDA_CHECK(cudaMemcpy(onb.data(), on.data, onn * 2, cudaMemcpyDeviceToHost));
        char ofn[96];
        std::snprintf(ofn, sizeof(ofn), "D:/ninfer-hip-migrate/build/gdn_on_%d_s%d.bin", gidx, gdn_decode_step);
        std::FILE* of = std::fopen(ofn, "wb");
        if (of) { std::fwrite(onb.data(), 2, onn, of); std::fclose(of); }
        // AMD port probe: per-layer scan output o (pre-gated_rmsnorm) for isolation.
        const std::size_t onn2 = static_cast<std::size_t>(o.numel());
        std::vector<std::uint16_t> ob(onn2);
        CUDA_CHECK(cudaMemcpy(ob.data(), o.data, onn2 * 2, cudaMemcpyDeviceToHost));
        std::snprintf(ofn, sizeof(ofn), "D:/ninfer-hip-migrate/build/gdn_o_%d_s%d.bin", gidx, gdn_decode_step);
        std::FILE* of2 = std::fopen(ofn, "wb");
        if (of2) { std::fwrite(ob.data(), 2, onn2, of2); std::fclose(of2); }
    }

    if (std::getenv("NINFER_GGUF_MIX_PROBE") != nullptr && gidx == 0) {
        const int n0 = static_cast<int>(on.ne[0]) * static_cast<int>(on.ne[1]);
        std::vector<std::uint16_t> o_h(static_cast<std::size_t>(
            o.ne[0] * o.ne[1])),
            on_h(static_cast<std::size_t>(on.ne[0] * on.ne[1])),
            x_h(static_cast<std::size_t>(x.ne[0] * x.ne[1]));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(o_h.data(), o.data, o_h.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(on_h.data(), on.data, on_h.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(x_h.data(), x.data, x_h.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* of = std::fopen("D:/ninfer-hip-migrate/build/gdn_o.bin", "wb");
        if (of) { std::fwrite(o_h.data(), 2, o_h.size(), of); std::fclose(of); }
        std::FILE* nf = std::fopen("D:/ninfer-hip-migrate/build/gdn_on.bin", "wb");
        if (nf) { std::fwrite(on_h.data(), 2, on_h.size(), nf); std::fclose(nf); }
        std::FILE* xf = std::fopen("D:/ninfer-hip-migrate/build/gdn_mixer_x.bin", "wb");
        if (xf) { std::fwrite(x_h.data(), 2, x_h.size(), xf); std::fclose(xf); }
        std::vector<std::uint16_t> z_h(static_cast<std::size_t>(z.numel()));
        CUDA_CHECK(cudaMemcpy(z_h.data(), z.data, z_h.size() * 2, cudaMemcpyDeviceToHost));
        std::FILE* zf = std::fopen("D:/ninfer-hip-migrate/build/gdn_z.bin", "wb");
        if (zf) { std::fwrite(z_h.data(), 2, z_h.size(), zf); std::fclose(zf); }
        std::fprintf(stderr, "[mix0] z dumped %zu elems\n", z_h.size());
        std::fprintf(stderr,
                     "[mix0] o=%p on=%p x=%p o0=%u,%u on0=%u,%u x0=%u,%u wq=%p o dumped=%zu on=%zu x=%zu\n",
                     static_cast<void*>(o.data), static_cast<void*>(on.data),
                     static_cast<void*>(x.data), static_cast<unsigned>(o_h[0]),
                     static_cast<unsigned>(o_h[1]), static_cast<unsigned>(on_h[0]),
                     static_cast<unsigned>(on_h[1]), static_cast<unsigned>(x_h[0]),
                     static_cast<unsigned>(x_h[1]), static_cast<const void*>(p.output.weight.qdata),
                     o_h.size(), on_h.size(), x_h.size());
        // 全 token 的 on（rmsnorm 后, linear_add 输入）
        std::FILE* nfull = std::fopen("D:/ninfer-hip-migrate/build/gdn_on_full.bin", "wb");
        if (nfull) {
            const std::size_t on_total = static_cast<std::size_t>(on.ne[0]) * on.ne[1] *
                                         on.ne[2];
            std::vector<std::uint16_t> on_full(on_total);
            CUDA_CHECK(cudaMemcpy(on_full.data(), on.data, on_total * 2, cudaMemcpyDeviceToHost));
            std::fwrite(on_full.data(), 2, on_full.size(), nfull);
            std::fclose(nfull);
        }
        // linear_add 前的 x (residual)
        std::FILE* xb = std::fopen("D:/ninfer-hip-migrate/build/gdn_mixer_x_before.bin", "wb");
        if (xb) { std::fwrite(x_h.data(), 2, x_h.size(), xb); std::fclose(xb); }
        std::fflush(stderr);
    }

    // Ablation hook (NINFER_SKIP_SSMOUT=1): drop the GDN output projection so its share of
    // the decode step can be separated from the projection/conv/recurrence work. Timing only.
    static const bool skip_ssmout = std::getenv("NINFER_SKIP_SSMOUT") != nullptr;
    if (!skip_ssmout) {
        ops::linear_add(on.view({dimension(config_.gdn->value_width()), T}), p.output.weight, x,
                        p.output.policy, work_, s);
    }
    if (std::getenv("NINFER_GDN_STATE_PROBE") != nullptr) {
        // per-layer x after the o_proj residual add, to isolate engine o_proj vs CPU.
        // The dump name carries the decode step so different steps do not overwrite it.
        cudaStreamSynchronize(s);
        const std::size_t nx = static_cast<std::size_t>(x.ne[0]) * x.ne[1];
        std::vector<std::uint16_t> xa(nx);
        CUDA_CHECK(cudaMemcpy(xa.data(), x.data, nx * 2, cudaMemcpyDeviceToHost));
        char xfn[96];
        std::snprintf(xfn, sizeof(xfn), "D:/ninfer-hip-migrate/build/gdn_x_after_%d_s%d.bin",
                      gidx, gdn_decode_step);
        std::FILE* xaf = std::fopen(xfn, "wb");
        if (xaf) { std::fwrite(xa.data(), 2, nx, xaf); std::fclose(xaf); }
        std::vector<std::uint16_t> xb(nx);
        CUDA_CHECK(cudaMemcpy(xb.data(), x.data, nx * 2, cudaMemcpyDeviceToHost));
        (void)xb;
    }
    if (std::getenv("NINFER_GGUF_MIX_PROBE") != nullptr && gidx == 0) {
        const std::size_t nx = static_cast<std::size_t>(x.ne[0]) * x.ne[1];
        std::vector<std::uint16_t> x_after(nx);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(x_after.data(), x.data, nx * 2, cudaMemcpyDeviceToHost));
        std::FILE* xaf = std::fopen("D:/ninfer-hip-migrate/build/gdn_mixer_x.bin", "wb");
        if (xaf) { std::fwrite(x_after.data(), 2, nx, xaf); std::fclose(xaf); }
        std::fprintf(stderr, "[mix0-after] x dumped %zu elems x0=%u,%u,%u,%u\n", nx,
                     static_cast<unsigned>(x_after[0]), static_cast<unsigned>(x_after[1]),
                     static_cast<unsigned>(x_after[2]), static_cast<unsigned>(x_after[3]));
        std::fflush(stderr);
    }
}

ops::SparseMoeHints TextContext::next_projection_hints(int layer) const {
    const auto next = static_cast<std::size_t>(layer) + 1;
    return next < parameters_.text.layers.size() ? parameters_.text.layers[next].projection_prefetch
                                                 : ops::SparseMoeHints{};
}

void TextContext::mlp_tail(const BlockParameters& weights, Tensor& x, Phase,
                           const ops::SparseMoeHints& hints) {
    Tensor h = workspace::post_mixer_hidden(work_, config_, x.ne[1]);
    ops::rmsnorm(x, weights.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
    ffn(h, weights.ffn, x, hints, work_, ctx_.stream);
}

// AMD port: decode per-mixer timing state (NINFER_PROFILE_DECODE=1)
struct DecMixTimer {
    bool on = false;
    LARGE_INTEGER freq = {};
    double attn = 0.0, gdn = 0.0, mlp = 0.0;
    int steps = 0;
};
static DecMixTimer g_dec_mixer_timer = [] {
    DecMixTimer d;
    // AMD port: the mixer timer needs device syncs, which destroys the CUDA graph and
    // inflates every number. Gate it on its own variable so NINFER_PROFILE_DECODE=1 can
    // collect the cudaEvent section table without touching the step structure.
    const char* e = std::getenv("NINFER_PROFILE_MIXER");
    d.on = e != nullptr && e[0] == '1';
    QueryPerformanceFrequency(&d.freq);
    return d;
}();

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    // B20 P0-2: NaN triage (NINFER_NAN_TRIAGE=1) scans the residual x after the mixer
    // and after the MLP each layer, printing the first NaN (tag = 10*layer + phase).
    static const bool nan_triage = [] {
        const char* e = std::getenv("NINFER_NAN_TRIAGE");
        return e != nullptr && e[0] == '1';
    }();
    const auto scan_x = [&](int layer, int phase) {
        if (nan_triage && prefill && x.data != nullptr) {
            ninfer::ninfer_nan_scan(x.data, static_cast<int>(x.ne[0]),
                                    static_cast<int>(x.ne[1]), 10 * layer + phase, ctx_.stream);
        }
    };
    // AMD port: NINFER_PROFILE_PREFILL=1 -> per-op cudaEvent timeline (no host sync
    // inside the loop; the D2H checksum syncs distort the op mix badly at large T).
    struct ProfState {
        bool on = false;
        std::vector<cudaEvent_t> events;
        std::vector<const char*> names;
        std::size_t cursor = 0;
        void mark(const char* name, cudaStream_t stream) {
            if (cursor >= events.size()) {
                events.push_back(cudaEvent_t{});
                names.push_back(nullptr);
                cudaEventCreate(&events.back());
            }
            if (events[cursor] == nullptr) { cudaEventCreate(&events[cursor]); }
            names[cursor] = name;
            cudaEventRecord(events[cursor++], stream);
        }
    };
    static ProfState prof = [] {
        ProfState s;
        const char* e = std::getenv("NINFER_PROFILE_PREFILL");
        s.on = e != nullptr && e[0] == '1';
        return s;
    }();
    static ProfState dprof = [] {
        ProfState s;
        const char* e = std::getenv("NINFER_PROFILE_DECODE");
        s.on = e != nullptr && e[0] == '1';
        return s;
    }();
    if (dprof.on && !prefill) {
        // Decode profile: aggregate per-segment ms across verify calls; print averages
        // every 64 calls (cudaEvents, no host sync inside the step).
        if (dprof.cursor + 6 >= dprof.events.size()) {
            dprof.events.resize(dprof.events.size() + 128);
            dprof.names.resize(dprof.events.size(), nullptr);
        }
        dprof.mark("embed", ctx_.stream);
    }
    if (prof.on && prefill) { prof.mark("embed", ctx_.stream); }
    debug_layer_checksum(prefill ? "embed" : "embed-d", -1, x, ctx_.stream);
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block  = parameters_.text.layers[layer];
        const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
        const auto compact = dimension(config_.compact_layer_indices[layer]);
        nvtx::ScopedRange layer_range(
            full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
                 : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        try {
            {
                nvtx::ScopedRange mixer_range(
                    full ? (prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
                         : (prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
                    full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
                auto scope = work_.scope();
                                if (prof.on && prefill) { prof.mark(full ? "mixer-attn" : "mixer-gdn", ctx_.stream); }
                if (dprof.on && !prefill) { dprof.mark(full ? "mixer-attn" : "mixer-gdn", ctx_.stream); }
                LARGE_INTEGER mt0 = {}, mt1 = {};
                if (g_dec_mixer_timer.on && !prefill) { CUDA_CHECK(cudaDeviceSynchronize()); QueryPerformanceCounter(&mt0); }
                static const bool skip_mixer = [] { const char* e = std::getenv("NINFER_SKIP_MIXER"); return e != nullptr && e[0] == '1'; }();
                if (!skip_mixer) {
                    if (full) {
                        attn_mix(block, x, compact, ph);
                    } else {
                        gdn_mix(block, x, compact, ph);
                    }
                }
                if (g_dec_mixer_timer.on && !prefill) {
                    CUDA_CHECK(cudaDeviceSynchronize());
                    QueryPerformanceCounter(&mt1);
                    if (full) { g_dec_mixer_timer.attn += static_cast<double>(mt1.QuadPart - mt0.QuadPart) * 1000.0 / static_cast<double>(g_dec_mixer_timer.freq.QuadPart); }
                    else { g_dec_mixer_timer.gdn += static_cast<double>(mt1.QuadPart - mt0.QuadPart) * 1000.0 / static_cast<double>(g_dec_mixer_timer.freq.QuadPart); }
                    if (++g_dec_mixer_timer.steps % 64 == 0) {
                        std::fprintf(stderr,
                                     "[prof-decode] per step: attn %.3f ms, gdn %.3f ms, mlp %.3f ms"
                                     " (layer-calls %d)\n",
                                     g_dec_mixer_timer.attn, g_dec_mixer_timer.gdn,
                                     g_dec_mixer_timer.mlp, g_dec_mixer_timer.steps);
                        std::fflush(stderr);
                        g_dec_mixer_timer.attn = 0.0;
                        g_dec_mixer_timer.gdn = 0.0;
                        g_dec_mixer_timer.mlp = 0.0;
                    }
                }
                if (prof.on && prefill) { prof.mark(full ? "mixer-attn-end" : "mixer-gdn-end", ctx_.stream); }
                if (dprof.on && !prefill) { dprof.mark(full ? "mixer-attn-end" : "mixer-gdn-end", ctx_.stream); }
                debug_layer_checksum(full ? "attn" : "gdn", static_cast<int>(layer), x,
                                     ctx_.stream);
                if (prefill && nan_triage_layer_capture_) {
                    const int bad = ninfer::ninfer_nan_count(
                        x.data, static_cast<int>(x.ne[0]), static_cast<int>(x.ne[1]),
                        ctx_.stream);
                    std::fprintf(stderr, "[layer-nan] layer=%d after-mixer(%s) nan=%d maxabs=%g\n",
                                 static_cast<int>(layer), full ? "attn" : "gdn", bad,
                                 ninfer::ninfer_nan_count_last_maxabs);
                    std::fflush(stderr);
                }
                if (prefill && std::getenv("NINFER_LAYER_PROBE") != nullptr && (layer < 4 || layer >= 61)) {
                    const int nx = static_cast<int>(x.ne[0]) * static_cast<int>(x.ne[1]);
                    std::vector<std::uint16_t> x_h(std::min(nx, 4));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(x_h.data(), x.data, x_h.size() * 2,
                                          cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "[layer%d] %s x0=%u,%u,%u,%u\n", static_cast<int>(layer),
                                 full ? "attn" : "gdn", static_cast<unsigned>(x_h[0]),
                                 static_cast<unsigned>(x_h[1]), static_cast<unsigned>(x_h[2]),
                                 static_cast<unsigned>(x_h[3]));
                    std::fflush(stderr);
                }
                scan_x(static_cast<int>(layer), 0);
            }
            {
                nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer
                                                : nvtx::Name::VerifyPostMixer,
                                        nvtx::Category::PostMixer, layer);
                auto scope = work_.scope();
                                if (prof.on && prefill) { prof.mark("mlp", ctx_.stream); }
                if (dprof.on && !prefill) { dprof.mark("mlp", ctx_.stream); }
                LARGE_INTEGER pt0 = {}, pt1 = {};
                if (g_dec_mixer_timer.on && !prefill) { CUDA_CHECK(cudaDeviceSynchronize()); QueryPerformanceCounter(&pt0); }
                static const bool skip_mlp = [] { const char* e = std::getenv("NINFER_SKIP_MLP"); return e != nullptr && e[0] == '1'; }();
                if (!skip_mlp) {
                    mlp_tail(block, x, ph, next_projection_hints(static_cast<int>(layer)));
                }
                debug_layer_checksum("mlp", static_cast<int>(layer), x, ctx_.stream);
                if (g_dec_mixer_timer.on && !prefill) {
                    CUDA_CHECK(cudaDeviceSynchronize());
                    QueryPerformanceCounter(&pt1);
                    g_dec_mixer_timer.mlp += static_cast<double>(pt1.QuadPart - pt0.QuadPart) * 1000.0 / static_cast<double>(g_dec_mixer_timer.freq.QuadPart);
                }
                if (prof.on && prefill) { prof.mark("mlp-end", ctx_.stream); }
                if (dprof.on && !prefill) { dprof.mark("mlp-end", ctx_.stream); }
                if (prefill && nan_triage_layer_capture_) {
                    const int bad = ninfer::ninfer_nan_count(
                        x.data, static_cast<int>(x.ne[0]), static_cast<int>(x.ne[1]),
                        ctx_.stream);
                    std::fprintf(stderr, "[layer-nan] layer=%d after-mlp nan=%d maxabs=%g\n",
                                 static_cast<int>(layer), bad,
                                 ninfer::ninfer_nan_count_last_maxabs);
                    std::fflush(stderr);
                }
                scan_x(static_cast<int>(layer), 1);
            }
            if constexpr (Tap::enabled) {
                tap.capture_layer(static_cast<int>(layer), x, ctx_.stream);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (prefill ? " prefill" : " verify") +
                                     " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
        }
    }
    if (dprof.on && !prefill) {
        dprof.mark("end", ctx_.stream);
        // aggregate into static accumulators; print every 64 calls
        struct DecAgg { double ms = 0.0; };
        static std::map<std::string, double> dec_acc;
        static int dec_calls = 0;
        static cudaEvent_t dec_anchor = nullptr;
        if (dec_anchor == nullptr) { cudaEventCreate(&dec_anchor); cudaEventRecord(dec_anchor, ctx_.stream); }
        ++dec_calls;
        for (std::size_t i = 1; i < dprof.cursor; ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, dprof.events[i - 1], dprof.events[i]) == cudaSuccess) {
                dec_acc[dprof.names[i - 1]] += ms;
            }
        }
        if (dec_calls % 64 == 0) {
            CUDA_CHECK(cudaEventSynchronize(dprof.events[dprof.cursor - 1]));
            float total = 0.0f;
            cudaEventElapsedTime(&total, dec_anchor, dprof.events[dprof.cursor - 1]);
            std::fprintf(stderr, "[prof-decode] %d steps, total %.1f ms (%.3f ms/step):\n",
                         dec_calls, total, total / 64.0);
            for (const auto& [name, ms] : dec_acc) {
                std::fprintf(stderr, "  %-16s %9.1f ms  (%.3f ms/step)\n", name.c_str(), ms, ms / 64.0);
            }
            std::fflush(stderr);
            dec_acc.clear();
            dec_anchor = dprof.events[dprof.cursor - 1];
            cudaEventRecord(dec_anchor, ctx_.stream);
        }
        dprof.cursor = 0;
    }
    if (prof.on && prefill) {
        prof.mark("end", ctx_.stream);
        cudaEventSynchronize(prof.events[prof.cursor - 1]);
        struct Agg { double ms = 0.0; int count = 0; };
        std::map<std::string, Agg> agg;
        double total = 0.0f;
        for (std::size_t i = 1; i < prof.cursor; ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, prof.events[i - 1], prof.events[i]) != cudaSuccess) { continue; }
            agg[prof.names[i - 1]].ms += ms;
            agg[prof.names[i - 1]].count += 1;
            total += ms;
        }
        std::fprintf(stderr, "[prof] prefill GPU timeline, total %.1f ms:\n", total);
        for (const auto& [name, a] : agg) {
            std::fprintf(stderr, "  %-16s %9.1f ms  x%d\n", name.c_str(), a.ms, a.count);
        }
        std::fflush(stderr);
        prof.cursor = 0;
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    // AMD DEBUG 2026-10-29: NINFER_NO_MTP_PROMPT=1 skips the always-on MTP prefill pass so the
    // target AR logits can be A/B tested for run-to-run determinism without MTP's own
    // (3-token) attention pass running first.
    static const bool mtp_prompt_off = [] {
        const char* e = std::getenv("NINFER_NO_MTP_PROMPT");
        return e != nullptr && e[0] == '1';
    }();
    const bool prepare_mtp_prompt =
        mtp_enabled() && io_.mtp.has_value() && !mtp_prompt_off;
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        // B20 P0-2: enable per-layer NaN counting on the final chunk only.
        nan_triage_layer_capture_ = is_last && std::getenv("NINFER_NAN_TRIAGE") != nullptr;
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace::text_prefill_roots(
                work_, config_, len, rope_axes,
                static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            // Resident window: visible counts address the compact KV slot space.
    const auto visible = static_cast<std::uint32_t>(base_i + t0 + len) -
                         g_ninfer_kv_window_base_tokens;
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);
            // Slot -> compact position map for the retained window (identity order today;
            // recall blocks will publish a permutation here instead).
            ScopedKeyPositionMap scoped_key_pos;
            Tensor key_pos_map;
            if (g_ninfer_kv_window_base_tokens != 0 && visible > 0) {
                key_pos_map = work_.alloc(DType::I32, {static_cast<std::int32_t>(visible)});
                ops::fill_i32_positions(key_pos_map, 0, s);
                scoped_key_pos.arm(&key_pos_map, static_cast<std::int32_t>(visible));
            }

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            // B20 P0-2: dump the rmsnorm INPUT **before** rmsnorm executes, so the value cannot
// have been overwritten by the rmsnorm output (aliases / in-place were the open risk).
            if (is_last) {
                const char* dir_pre = std::getenv("NINFER_DUMP_LAYERS");
                // B20 P0-2: count NaN at the SAME point as the dump (pre-rmsnorm).
                // Compare with the in-loop per-layer counts: equal => the dump offset
                // reads different memory; different => x was modified after run_layers.
                // AMD port: NaN pre-scan is a kernel + sync per step; it used to run on EVERY
                // decode step and both cost time and polluted the chat stream. Debug only.
                if (std::getenv("NINFER_DBG_PTR") != nullptr) {
                    const int pre_nan = ninfer::ninfer_nan_count(
                        x.data, static_cast<int>(x.ne[0]), static_cast<int>(x.ne[1]),
                        ctx_.stream);
                    std::fprintf(stderr, "[rmsnorm-ptr] x.data=%p x.bytes=%zu "
                                         "pre-rmsnorm-nan=%d ne=[%d,%d]\n",
                                 x.data, x.bytes(), pre_nan, static_cast<int>(x.ne[0]),
                                 static_cast<int>(x.ne[1]));
                    std::fflush(stderr);
                }
                // B20 P0-2: print the pointers to prove/disprove in-place aliasing.
                // Gated behind NINFER_DUMP_LAYERS (not enabled by default).
                const char* dir_rms = std::getenv("NINFER_DUMP_LAYERS");
                if (dir_rms != nullptr) {
                    std::fprintf(stderr,
                                 "[rmsnorm-ptr] x.data=%p x.bytes=%zu | prefill_hidden_.data=%p\n",
                                 x.data, x.bytes(),
                                 static_cast<const void*>(prefill_hidden_.data));
                    std::fflush(stderr);
                }
                if (dir_pre != nullptr) {
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    const std::size_t hdim =
                        static_cast<std::size_t>(dimension(config_.hidden_size));
                    const std::uint16_t* xlast =
                        static_cast<const std::uint16_t*>(x.data) + (len - 1) * hdim;
                    std::vector<std::uint16_t> hxin(hdim);
                    CUDA_CHECK(cudaMemcpy(hxin.data(), xlast, hdim * sizeof(std::uint16_t),
                                          cudaMemcpyDeviceToHost));
                    std::FILE* f0 =
                        std::fopen((std::string(dir_pre) + "_rmsnorm_in.bin").c_str(), "wb");
                    if (f0 != nullptr) { std::fwrite(hxin.data(), sizeof(std::uint16_t), hdim, f0); std::fclose(f0); }
                }
            }
            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
            ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, s);
            debug_layer_checksum("final-pre", 98, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                // B20 P0-2: dump the final prefill hidden (5120 bf16) to check if the
                // int8 path drifts hidden values large before lm_head.
                if (const char* dir = std::getenv("NINFER_DUMP_LAYERS")) {
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    const std::size_t hdim = static_cast<std::size_t>(dimension(config_.hidden_size));
                    std::vector<std::uint16_t> hxf(hdim);
                    CUDA_CHECK(cudaMemcpy(hxf.data(), last_xf.data, hdim * sizeof(std::uint16_t),
                                          cudaMemcpyDeviceToHost));
                    std::FILE* fx =
                        std::fopen((std::string(dir) + "_final_hidden.bin").c_str(), "wb");
                    if (fx != nullptr) {
                        std::fwrite(hxf.data(), sizeof(std::uint16_t), hdim, fx);
                        std::fclose(fx);
                    }
                }
                Tensor logits  = matrix_window(io_.logits, 1);
                project(last_xf, *lm_head_, logits, work_, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count),
                                sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count), s);
                }
                if (std::getenv("NINFER_TOKEN_PROBE") != nullptr) {
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    std::int32_t tokv = -1;
                    CUDA_CHECK(cudaMemcpy(&tokv, io_.token.data, sizeof(std::int32_t),
                                          cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "[tokenprobe] prefill sampled token = %d\n", tokv);
                    std::fflush(stderr);
                }
                // AMD DEBUG: dump full logits for offline comparison with the reference runtime.
                // B20 P0-2: unconditional one-shot dump of the final prefill logits, to
                // confirm whether the "!" collapse is a logits-degeneration (finite but
                // flat/biased) rather than NaN. Removed after triage.
                {
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    const std::size_t vocab = static_cast<std::size_t>(
                        dimension(parameters_.model.resources().public_token_count));
                    std::vector<std::uint16_t> hlogits(vocab);
                    CUDA_CHECK(cudaMemcpy(hlogits.data(), logits.data,
                                          vocab * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
                    if (const char* dir = std::getenv("NINFER_DUMP_LAYERS")) {
                        std::FILE* f =
                            std::fopen((std::string(dir) + "_prefill_logits.bin").c_str(), "wb");
                        if (f != nullptr) {
                            std::fwrite(hlogits.data(), sizeof(std::uint16_t), vocab, f);
                            std::fclose(f);
                        }
                    }
                }
                if (std::getenv("NINFER_NAN_TRIAGE") != nullptr) {
                    ninfer::ninfer_nan_triage_report();
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings =
                        work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_5::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden =
                            work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i) -
                            g_ninfer_kv_window_base_tokens;
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        ScopedKeyPositionMap scoped_key_pos;
                        Tensor key_pos_map;
                        if (g_ninfer_kv_window_base_tokens != 0 && ar_visible > 0) {
                            key_pos_map =
                                work_.alloc(DType::I32, {static_cast<std::int32_t>(ar_visible)});
                            ops::fill_i32_positions(key_pos_map, 0, s);
                            scoped_key_pos.arm(&key_pos_map,
                                               static_cast<std::int32_t>(ar_visible));
                        }
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    // The prompt is complete: snapshot the query summary so the first decode step ranks archived
    // pages against the question instead of whatever chunk happens to be in flight.
    if (finalize_at_end && t0 >= T) {
        if (auto* retrieval = ninfer::models::qwen3_5::detail::KVRetrieval::instance();
            retrieval != nullptr) {
            retrieval->freeze_query(s);
        }
    }

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, sink,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_5::execution





