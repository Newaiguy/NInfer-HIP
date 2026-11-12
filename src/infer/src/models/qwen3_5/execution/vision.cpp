#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/vision.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/nvtx.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/scale.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <set>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <fstream>

namespace ninfer::models::qwen3_5::execution {
namespace {

struct VisionStreamEntry {
    Weight* target = nullptr;
    std::vector<std::uint16_t> host;
};

// The packaged vision tower has no final LayerNorm at all, while the reference checkpoint applies
// one after the last block. Without it the features leave the tower roughly twenty times too wide
// and carry huge negative outliers, which the language model turns into a flat description. The
// two pointers below are filled from an override manifest when one supplies vision/post_ln_*.
void* g_post_ln_weight       = nullptr;
void* g_post_ln_bias         = nullptr;

std::vector<VisionStreamEntry>& vision_stream_entries() {
    static std::vector<VisionStreamEntry> entries;
    return entries;
}

void*& vision_stream_staging() {
    static void* staging = nullptr;
    return staging;
}

std::size_t& vision_stream_staging_bytes() {
    static std::size_t bytes = 0;
    return bytes;
}

// Streams one bf16 tensor into a single reused device buffer: a bf16 vision tower then costs about
// one tensor of VRAM (47 MB) instead of the whole 0.91 GB weight set.
void ensure_vision_weight_upload(Weight& weight) {
    for (auto& entry : vision_stream_entries()) {
        if (entry.target != &weight || entry.host.empty()) { continue; }
        const std::size_t bytes = entry.host.size() * sizeof(std::uint16_t);
        // The staging buffer is allocated once and never released: freeing device memory that the
        // compute stream may still be reading wedges the driver.
        if (vision_stream_staging() == nullptr) {
            void* fresh = nullptr;
            constexpr std::size_t kStagingCapacity = 96U * 1024U * 1024U;
            if (cudaMalloc(&fresh, kStagingCapacity) != cudaSuccess) { return; }
            vision_stream_staging()       = fresh;
            vision_stream_staging_bytes() = kStagingCapacity;
        }
        if (vision_stream_staging_bytes() < bytes) { return; }
        // The previous layer still reads this buffer on the compute stream, so the overwrite has to
        // wait for it; the tower runs once per image, so a device barrier here is affordable.
        (void)cudaDeviceSynchronize();
        void* device = vision_stream_staging();
        (void)cudaMemcpy(device, entry.host.data(), bytes, cudaMemcpyHostToDevice);
        weight.payload = device;
        weight.qdata   = device;
        return;
    }
}


std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("Vision ") + label +
                                    " alignment must be a power of two");
    }
    return checked_add(value, alignment - 1, label) & ~(alignment - 1);
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct VisionWorkspaceLayout {
    TensorRegion position_ids;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion merger_hidden;
    LayoutRegion patch_scratch, qkv_scratch, projection_scratch;
    LayoutRegion up_scratch, down_scratch, merger_first_scratch, merger_second_scratch;
    std::size_t bytes = 0;
};

TensorRegion alias_tensor(const TensorRegion& storage, DType dtype,
                          std::initializer_list<std::int32_t> shape, const char* label) {
    Tensor tensor(nullptr, dtype, shape);
    if (tensor.bytes() > storage.region.bytes) {
        throw std::logic_error(std::string("Vision ") + label +
                               " does not fit its aliased storage");
    }
    TensorRegion out;
    out.region = LayoutRegion{storage.region.offset, tensor.bytes(), storage.region.alignment};
    out.dtype  = dtype;
    std::copy(shape.begin(), shape.end(), out.shape.begin());
    return out;
}

VisionWorkspaceLayout build_workspace_layout(const VisionConfig& config,
                                             const VisionParameters& parameters,
                                             std::size_t patches64, std::size_t tokens64,
                                             std::size_t handoff_offset_bytes) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 != checked_mul(tokens64,
                                 dimension(std::uint64_t(config.spatial_merge_size) *
                                           config.spatial_merge_size),
                                 "patch/token relation")) {
        throw std::invalid_argument(
            "Vision workspace requires the configured positive patch/token ratio");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    const auto patches = static_cast<std::int32_t>(patches64);
    const auto tokens  = static_cast<std::int32_t>(tokens64);

    const auto capacity = [](const LinearParameters& p, std::int32_t first, std::int32_t last) {
        return ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                    p.policy, first, last);
    };
    const auto minimum_patches =
        dimension(std::uint64_t(config.spatial_merge_size) * config.spatial_merge_size);
    std::size_t qkv_bytes = 0, projection_bytes = 0, up_bytes = 0, down_bytes = 0;
    for (const auto& layer : parameters.layers) {
        qkv_bytes = std::max(qkv_bytes, capacity(layer.qkv, minimum_patches, patches));
        projection_bytes =
            std::max(projection_bytes, capacity(layer.output, minimum_patches, patches));
        up_bytes   = std::max(up_bytes, capacity(layer.fc1, minimum_patches, patches));
        down_bytes = std::max(down_bytes, capacity(layer.fc2, minimum_patches, patches));
    }
    LayoutBuilder builder;
    const auto scratch = [&](std::size_t bytes, const char* label) {
        auto scope = builder.scope();
        // A borrowed WorkspaceArena needs backing even when this Op uses no scratch.
        return builder.add(std::max(std::size_t{1}, bytes), kWorkspaceAlignment, label);
    };
    VisionWorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.x = add(DType::BF16, {dimension(config.hidden_size), patches}, "vision residual");
    {
        auto position_lifetime = builder.scope();
        out.position_ids       = add(DType::I32, {patches, 2}, "vision position ids");
        {
            auto patch_scope = builder.scope();
            out.patch_bf16 =
                add(DType::BF16, {dimension(config.patch_width()), patches}, "vision BF16 patches");
            out.patch_scratch =
                scratch(capacity(parameters.patch_embedding, minimum_patches, patches),
                        "patch projection scratch");
        }
        {
            auto position_scope = builder.scope();
            out.pos_indices     = add(DType::I32, {4, patches}, "vision position indices");
            out.pos_weights     = add(DType::FP32, {4, patches}, "vision position weights");
        }
        {
            auto attention_scope = builder.scope();
            out.qkv = add(DType::BF16, {3 * dimension(config.hidden_size), patches}, "vision QKV");
            out.attention_norm     = add(DType::BF16, {dimension(config.hidden_size), patches},
                                         "vision attention norm/attended");
            out.attended           = out.attention_norm;
            out.qkv_scratch        = scratch(qkv_bytes, "QKV scratch");
            out.projection_scratch = scratch(projection_bytes, "attention output scratch");
            out.projected =
                alias_tensor(out.qkv, DType::BF16, {dimension(config.hidden_size), patches},
                             "attention projection output");
        }
        {
            auto mlp_scope = builder.scope();
            out.mlp_up =
                add(DType::BF16, {dimension(config.intermediate_size), patches}, "vision MLP up");
            out.mlp_norm =
                add(DType::BF16, {dimension(config.hidden_size), patches}, "vision MLP norm/down");
            out.mlp_down     = out.mlp_norm;
            out.up_scratch   = scratch(up_bytes, "MLP up scratch");
            out.down_scratch = scratch(down_bytes, "MLP down scratch");
        }
    }
    {
        auto merger_scope = builder.scope();
        out.normalized =
            add(DType::BF16, {dimension(config.hidden_size), patches}, "vision merger norm");
        out.merger_first_scratch =
            scratch(capacity(parameters.merger_fc1, 1, tokens), "merger first scratch");
        out.merger_hidden = alias_tensor(
            out.x, DType::BF16, {dimension(config.merger_width()), tokens}, "merger hidden");
    }
    out.bytes                = builder.finish(1, "vision workspace");
    const auto final_scratch = std::max(std::size_t{1}, capacity(parameters.merger_fc2, 1, tokens));
    {
        LayoutBuilder finish;
        (void)finish.add(handoff_offset_bytes, kWorkspaceAlignment, "handoff start");
        (void)finish.add(
            checked_mul(checked_mul(parameters.merger_fc2.weight.n, tokens64, "output elements"), 2,
                        "output bytes"),
            1, "handoff");
        out.merger_second_scratch =
            finish.add(final_scratch, kWorkspaceAlignment, "merger second scratch");
        out.bytes = std::max(out.bytes, finish.finish(1, "merger call"));
    }
    return out;
}

std::size_t output_handoff_bytes(std::int32_t output_hidden, std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output handoff extent must fit positive int32");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(DType::BF16, {output_hidden, static_cast<std::int32_t>(merged_tokens)},
                            kWorkspaceAlignment, "Vision item output handoff");
    return layout.finish(kWorkspaceAlignment, "Vision item output handoff layout");
}

std::size_t merger_hidden_bytes(const VisionConfig& config, std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision merger hidden extent must fit positive int32");
    }
    Tensor tensor(nullptr, DType::BF16,
                  {dimension(config.merger_width()), static_cast<std::int32_t>(merged_tokens)});
    return tensor.bytes();
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const Parameters& parameters)
    : ctx_(ctx), config_(parameters.model.config().vision.value()),
      parameters_(parameters.vision.value()) {}

std::size_t VisionContext::workspace_bytes(const VisionConfig& config,
                                           const VisionParameters& parameters, std::size_t patches,
                                           std::size_t merged_tokens,
                                           const VisionWorkspacePlan& plan) {
    return build_workspace_layout(config, parameters, patches, merged_tokens,
                                  plan.handoff_offset_bytes)
        .bytes;
}

VisionWorkspacePlan VisionContext::plan_workspace(const VisionConfig& config,
                                                  const VisionParameters& parameters,
                                                  std::uint32_t max_merged_tokens,
                                                  std::size_t general_capacity_bytes) {
    if (max_merged_tokens == 0 || general_capacity_bytes == 0) {
        throw std::invalid_argument("Vision workspace extents must be positive");
    }
    VisionWorkspacePlan out;
    out.output_hidden          = parameters.merger_fc2.weight.n;
    out.max_merged_tokens      = max_merged_tokens;
    out.general_capacity_bytes = general_capacity_bytes;
    out.handoff_offset_bytes =
        align_up(std::max(general_capacity_bytes, merger_hidden_bytes(config, max_merged_tokens)),
                 kWorkspaceAlignment, "handoff offset");
    out.handoff_capacity_bytes = output_handoff_bytes(out.output_hidden, max_merged_tokens);
    out.encode_peak_bytes =
        build_workspace_layout(
            config, parameters,
            checked_mul(max_merged_tokens,
                        std::uint64_t(config.spatial_merge_size) * config.spatial_merge_size,
                        "capacity patch count"),
            max_merged_tokens, out.handoff_offset_bytes)
            .bytes;
    out.capacity_bytes = std::max(
        out.encode_peak_bytes,
        checked_add(out.handoff_offset_bytes, out.handoff_capacity_bytes, "workspace capacity"));
    return out;
}

Tensor VisionContext::bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                  std::size_t merged_tokens) {
    if (backing.data == nullptr || backing.bytes < plan.capacity_bytes || merged_tokens == 0 ||
        merged_tokens > plan.max_merged_tokens) {
        throw std::invalid_argument("Vision output binding exceeds its workspace plan");
    }
    const std::size_t bytes = output_handoff_bytes(plan.output_hidden, merged_tokens);
    if (bytes > plan.handoff_capacity_bytes) {
        throw std::logic_error("Vision output binding exceeds its handoff region");
    }
    TensorRegion region;
    region.region = LayoutRegion{plan.handoff_offset_bytes, bytes, kWorkspaceAlignment};
    region.dtype  = DType::BF16;
    region.shape  = {plan.output_hidden, static_cast<std::int32_t>(merged_tokens), 1, 1};
    return region.bind(backing);
}

void VisionContext::encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                           const VisionWorkspacePlan& plan) const {
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const qwen3_5::VisionItemControl& control = *item.control;
    const auto patches64                      = control.patch_count;
    const auto tokens64                       = control.merged_count;
    nvtx::ScopedRange encode_range(nvtx::Name::VisionEncode, nvtx::Category::Vision,
                                   static_cast<std::uint64_t>(patches64));
    if (std::getenv("NINFER_VISION_TRACE") != nullptr) {
        std::fprintf(stderr, "[vision-ctl] patch=%lld merged=%lld segment=%d heads=%d hidden=%d\n",
                     static_cast<long long>(patches64), static_cast<long long>(tokens64),
                     static_cast<int>(control.segment_length),
                     static_cast<int>(config_.num_heads), static_cast<int>(config_.hidden_size));
        std::fflush(stderr);
    }
    if (item.patches.size() !=
        checked_mul(patches64, dimension(config_.patch_width()), "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != parameters_.merger_fc2.weight.n ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V]");
    }
    const Tensor planned_output = bind_output(backing, plan, tokens64);
    if (output.data != planned_output.data || output.bytes() != planned_output.bytes()) {
        throw std::invalid_argument("Vision output does not name the planned handoff region");
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(
        config_, parameters_, patches64, tokens64, plan.handoff_offset_bytes);
    if (layout.bytes > plan.encode_peak_bytes || backing.bytes < plan.capacity_bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;
    static bool vision_override_applied = false;
    if (!vision_override_applied && !parameters_.layers.empty()) {
        vision_override_applied = true;
        if (const char* root = std::getenv("NINFER_VISION_WEIGHTS"); root != nullptr) {
            const std::string manifest_path = std::string(root) + "\\manifest.tsv";
            std::vector<std::string> manifest_lines;
            {
                std::ifstream manifest_stream(manifest_path);
                std::string manifest_line;
                while (std::getline(manifest_stream, manifest_line)) {
                    manifest_lines.push_back(manifest_line);
                }
            }
            std::fprintf(stderr, "[vision-ovr] manifest lines=%zu path=%s\n", manifest_lines.size(),
                         manifest_path.c_str());
            std::fflush(stderr);
            int applied = 0;
            if (!manifest_lines.empty()) {
                char line[512];
                for (const std::string& source_line : manifest_lines) {
                    std::snprintf(line, sizeof(line), "%s", source_line.c_str());
                    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') { continue; }
                    char patch_file[256] = {0};
                    if (std::strncmp(line, "vision/patch_embedding", 22) == 0 &&
                        std::sscanf(line, "vision/patch_embedding %255s", patch_file) == 1) {
                        const Weight* patch = &parameters_.patch_embedding.weight;
                        const std::size_t rows = static_cast<std::size_t>(patch->shape[0]);
                        const std::size_t cols = static_cast<std::size_t>(patch->shape[1]);
                        const std::size_t want = rows * cols;
                        const std::string path = std::string(root) + "\\" + patch_file;
                        std::FILE* fp = std::fopen(path.c_str(), "rb");
                        if (fp == nullptr) { continue; }
                        std::vector<std::uint16_t> host(want);
                        const std::size_t got = std::fread(host.data(), sizeof(std::uint16_t), host.size(), fp);
                        std::fclose(fp);
                        if (got != host.size()) { continue; }
                        // The binder hands back (in, out) for this tensor while every other layer is
                        // (out, in); a transposed matrix turns the colour channels into noise.
                        const std::size_t out_dim =
                            static_cast<std::size_t>(parameters_.patch_embedding_bias.ne[0]);
                        if (cols != out_dim && rows == out_dim) {
                            std::vector<std::uint16_t> flipped(want);
                            for (std::size_t r = 0; r < rows; ++r) {
                                for (std::size_t c = 0; c < cols; ++c) {
                                    flipped[c * rows + r] = host[r * cols + c];
                                }
                            }
                            host.swap(flipped);
                        }
                        Weight* writable          = const_cast<Weight*>(patch);
                        writable->qtype           = QType::BF16;
                        writable->payload         = nullptr;
                        writable->payload_bytes   = want * sizeof(std::uint16_t);
                        writable->qdata           = nullptr;
                        writable->qhigh           = nullptr;
                        writable->scales          = nullptr;
                        writable->n               = static_cast<std::int32_t>(rows);
                        writable->k               = static_cast<std::int32_t>(cols);
                        writable->group           = 0;
                        writable->padded_shape[0] = static_cast<std::int32_t>(rows);
                        writable->padded_shape[1] = static_cast<std::int32_t>(cols);
                        vision_stream_entries().push_back(VisionStreamEntry{writable, std::move(host)});
                        ++applied;
                        continue;
                    }
                    char post_file[256] = {0};
                    char post_tag[64]   = {0};
                    if (std::strncmp(line, "vision/post_ln_", 15) == 0 &&
                        std::sscanf(line, "vision/post_ln_%63s %255s", post_tag, post_file) == 2) {
                        const std::size_t count =
                            static_cast<std::size_t>(parameters_.merger_norm.weight.ne[0]);
                        const std::string path = std::string(root) + "\\" + post_file;
                        std::FILE* fp          = std::fopen(path.c_str(), "rb");
                        if (fp == nullptr) { continue; }
                        std::vector<std::uint16_t> host(count);
                        const std::size_t got =
                            std::fread(host.data(), sizeof(std::uint16_t), count, fp);
                        std::fclose(fp);
                        if (got != count) { continue; }
                        // The artifact binder uses these scales verbatim and the export carries the same
                        // numbers, so no delta offset is applied. Adding 1.0 here washed every image out
                        // to a uniform grey ("silver") because the stored gammas are already final.
                        void* device = nullptr;
                        if (cudaMalloc(&device, count * sizeof(std::uint16_t)) != cudaSuccess) {
                            continue;
                        }
                        (void)cudaMemcpy(device, host.data(), count * sizeof(std::uint16_t),
                                         cudaMemcpyHostToDevice);
                        if (std::strcmp(post_tag, "weight") == 0) {
                            g_post_ln_weight = device;
                        } else {
                            g_post_ln_bias = device;
                        }
                        ++applied;
                        continue;
                    }
                    char merger_file[256] = {0};
                    if (std::strncmp(line, "vision/merger/", 14) == 0 &&
                        std::sscanf(line, "vision/merger/%*63s %255s", merger_file) == 1) {
                        const Weight* merged = std::strstr(line, "fc1") != nullptr
                                                   ? &parameters_.merger_fc1.weight
                                                   : &parameters_.merger_fc2.weight;
                        const std::size_t rows = static_cast<std::size_t>(merged->shape[0]);
                        const std::size_t cols = static_cast<std::size_t>(merged->shape[1]);
                        const std::size_t want = rows * cols;
                        const std::string path = std::string(root) + "\\" + merger_file;
                        std::FILE* fp = std::fopen(path.c_str(), "rb");
                        if (fp == nullptr) { continue; }
                        std::vector<std::uint16_t> host(want);
                        const std::size_t got = std::fread(host.data(), sizeof(std::uint16_t), host.size(), fp);
                        std::fclose(fp);
                        if (got != host.size()) { continue; }
                        void* device = nullptr;
                        Weight* writable          = const_cast<Weight*>(merged);
                        writable->qtype           = QType::BF16;
                        writable->payload         = nullptr;
                        writable->payload_bytes   = want * sizeof(std::uint16_t);
                        writable->qdata           = nullptr;
                        writable->qhigh           = nullptr;
                        writable->scales          = nullptr;
                        writable->n               = static_cast<std::int32_t>(rows);
                        writable->k               = static_cast<std::int32_t>(cols);
                        writable->group           = 0;
                        writable->padded_shape[0] = static_cast<std::int32_t>(rows);
                        writable->padded_shape[1] = static_cast<std::int32_t>(cols);
                    vision_stream_entries().push_back(VisionStreamEntry{writable, std::move(host)});
                        ++applied;
                        continue;
                    }
                    unsigned layer_index = 0;
                    char suffix[64] = {0};
                    char fname[256] = {0};
                    if (std::sscanf(line, "vision/layers/%u/%63s %255s", &layer_index, suffix, fname) != 3) {
                        continue;
                    }
                    if (layer_index >= parameters_.layers.size()) { continue; }
                    auto& block = parameters_.layers[layer_index];
                    {
                        // Vision norms are stored as small deltas in the reference: the usable scale is
                        // one plus the stored weight. Biases are copied verbatim.
                        const Tensor* scale = nullptr;
                        const Tensor* shift = nullptr;
                        if (std::strcmp(suffix, "norm1_weight") == 0) { scale = &block.norm1.weight; }
                        else if (std::strcmp(suffix, "norm1_bias") == 0) { shift = &block.norm1.bias; }
                        else if (std::strcmp(suffix, "norm2_weight") == 0) { scale = &block.norm2.weight; }
                        else if (std::strcmp(suffix, "norm2_bias") == 0) { shift = &block.norm2.bias; }
                        else if (std::strcmp(suffix, "attention/qkv_bias") == 0) { shift = &block.qkv_bias; }
                        else if (std::strcmp(suffix, "attention/output_bias") == 0) { shift = &block.output_bias; }
                        else if (std::strcmp(suffix, "mlp/fc1_bias") == 0) { shift = &block.fc1_bias; }
                        else if (std::strcmp(suffix, "mlp/fc2_bias") == 0) { shift = &block.fc2_bias; }
                        const Tensor* scalar = scale != nullptr ? scale : shift;
                        if (scalar != nullptr) {
                            const std::size_t count = static_cast<std::size_t>(scalar->ne[0]);
                            const std::string path = std::string(root) + "\\" + fname;
                            std::FILE* fp = std::fopen(path.c_str(), "rb");
                            if (fp == nullptr) { continue; }
                            std::vector<std::uint16_t> host(count);
                            const std::size_t got = std::fread(host.data(), sizeof(std::uint16_t), count, fp);
                            std::fclose(fp);
                            if (got != count) { continue; }
                            // Scales are copied verbatim: the export matches what the artifact binder
                            // hands the kernel, and the "+1 delta" reading turned every image silver.
                            (void)cudaMemcpy(const_cast<void*>(scalar->data), host.data(),
                                             count * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
                            ++applied;
                            continue;
                        }
                    }
                    const Weight* target = nullptr;
                    if (std::strcmp(suffix, "attention/qkv") == 0) { target = &block.qkv.weight; }
                    else if (std::strcmp(suffix, "attention/output") == 0) { target = &block.output.weight; }
                    else if (std::strcmp(suffix, "mlp/fc1") == 0) { target = &block.fc1.weight; }
                    else if (std::strcmp(suffix, "mlp/fc2") == 0) { target = &block.fc2.weight; }
                    if (target == nullptr) { continue; }
                    const std::size_t rows = static_cast<std::size_t>(target->shape[0]);
                    const std::size_t cols = static_cast<std::size_t>(target->shape[1]);
                    const std::size_t want = rows * cols;
                    const std::string path = std::string(root) + "\\" + fname;
                    std::FILE* fp = std::fopen(path.c_str(), "rb");
                    if (fp == nullptr) { continue; }
                    std::vector<std::uint16_t> host(want);
                    const std::size_t got = std::fread(host.data(), sizeof(std::uint16_t), host.size(), fp);
                    std::fclose(fp);
                    if (got != host.size()) { continue; }
                    Weight* writable          = const_cast<Weight*>(target);
                    writable->qtype           = QType::BF16;
                    writable->payload         = nullptr;
                    writable->payload_bytes   = want * sizeof(std::uint16_t);
                    writable->qdata           = nullptr;
                    writable->qhigh           = nullptr;
                    writable->scales          = nullptr;
                    writable->n               = static_cast<std::int32_t>(rows);
                    writable->k               = static_cast<std::int32_t>(cols);
                    writable->group           = 0;
                    writable->padded_shape[0] = static_cast<std::int32_t>(rows);
                    writable->padded_shape[1] = static_cast<std::int32_t>(cols);
                    vision_stream_entries().push_back(VisionStreamEntry{writable, std::move(host)});
                    ++applied;
                }
            }
            std::fprintf(stderr, "[vision-ovr] applied %d tensors from %s\n", applied, root);
            std::fflush(stderr);
        }
    }
    // The workspace is reused across requests, and the vision attention tiles read a padded tail
    // past the real token count. Leftover bytes from a previous image would land in those lanes and
    // poison the layer 0 softmax (observed as NaNs and drifting answers on the second request), so
    // the encode starts from a zeroed arena every time.
    (void)cudaMemsetAsync(backing.data, 0, layout.bytes, stream);

    const auto project = [&](const Tensor& x, const LinearParameters& p, Tensor& out,
                             const LayoutRegion& region) {
        WorkspaceArena scratch(region.bind(backing));
        ensure_vision_weight_upload(const_cast<Weight&>(p.weight));
        if (p.weight.payload == nullptr) {
            throw std::runtime_error("vision linear: weight payload is not resident");
        }
        ops::linear(x, p.weight, out, p.policy, scratch, stream);
        if (std::getenv("NINFER_VISION_TRACE") != nullptr) {
            static int weight_probe = 0;
            if (weight_probe < 6) {
                ++weight_probe;
                std::fprintf(stderr, "[vision-wp] qtype=%d n=%d k=%d payload=%p pad=%d,%d\n",
                             static_cast<int>(p.weight.qtype), static_cast<int>(p.weight.n),
                             static_cast<int>(p.weight.k), p.weight.payload,
                             static_cast<int>(p.weight.padded_shape[0]),
                             static_cast<int>(p.weight.padded_shape[1]));
                std::fflush(stderr);
            }
        }
    };
    const auto nan_scan = [&](const Tensor& value, const char* tag) {
        if (std::getenv("NINFER_VISION_NAN") == nullptr) { return; }
        (void)cudaStreamSynchronize(stream);
        const std::size_t count = static_cast<std::size_t>(value.ne[0]) * static_cast<std::size_t>(value.ne[1]);
        std::vector<std::uint16_t> host(count);
        (void)cudaMemcpy(host.data(), value.data, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        std::size_t bad = 0; double total = 0.0; std::size_t first_bad = count;
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t w = static_cast<std::uint32_t>(host[i]) << 16; float f = 0.0F;
            std::memcpy(&f, &w, sizeof(float));
            if (!std::isfinite(f)) { if (bad == 0) { first_bad = i; } ++bad; } else { total += static_cast<double>(f); }
        }
        std::fprintf(stderr, "[vision-stage] %s nonfinite=%zu/%zu first_token=%zu mean=%.4f\n", tag, bad, count,
                     first_bad / static_cast<std::size_t>(config_.hidden_size),
                     bad == count ? 0.0 : total / static_cast<double>(count - bad));
        std::fflush(stderr);
    };
    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor x            = layout.x.bind(backing);
    Tensor patch_bf16   = layout.patch_bf16.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    {
        nvtx::ScopedRange patch_range(nvtx::Name::VisionPatchEmbedding, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(patches64));
        copy_host(control.position_ids.data(), position_ids, stream);
        copy_host(item.patches.data(), patch_bf16, stream);
        if (const char* dump_path = std::getenv("NINFER_VISION_DUMP"); dump_path != nullptr) {
            (void)cudaStreamSynchronize(stream);
            const std::size_t count = static_cast<std::size_t>(patch_bf16.ne[0]) *
                                      static_cast<std::size_t>(patch_bf16.ne[1]);
            std::vector<std::uint16_t> raw_patches(count);
            (void)cudaMemcpy(raw_patches.data(), patch_bf16.data, count * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost);
            std::ofstream patch_file(std::string(dump_path) + "\\patches_in.f32", std::ios::binary);
            for (const std::uint16_t raw16 : raw_patches) {
                const std::uint32_t bits = static_cast<std::uint32_t>(raw16) << 16;
                float value = 0.0F;
                std::memcpy(&value, &bits, sizeof(value));
                patch_file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
        if (std::getenv("NINFER_VISION_NAN") != nullptr) {
            (void)cudaStreamSynchronize(stream);
            const std::size_t np = static_cast<std::size_t>(patch_bf16.ne[0]) * static_cast<std::size_t>(patch_bf16.ne[1]);
            std::vector<std::uint16_t> hp(np);
            (void)cudaMemcpy(hp.data(), patch_bf16.data, np * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
            std::size_t bad = 0;
            for (std::size_t i = 0; i < np; ++i) {
                std::uint32_t w = static_cast<std::uint32_t>(hp[i]) << 16; float f = 0.0F;
                std::memcpy(&f, &w, sizeof(float));
                if (!std::isfinite(f)) { ++bad; }
            }
            std::fprintf(stderr, "[vision-in] patch_nonfinite=%zu/%zu\n", bad, np); std::fflush(stderr);
        }
        project(patch_bf16, parameters_.patch_embedding, x, layout.patch_scratch);
        ops::add_bias(parameters_.patch_embedding_bias, x, stream);
        // The payload carries two identical frames for a still image, so the temporal axis doubles the
        // convolution sum. Averaging the frames back keeps the embedding on the scale the tower was
        // trained at.
        if (const char* emb_env = std::getenv("NINFER_VISION_EMB_SCALE"); emb_env != nullptr) {
            const float emb_factor = std::strtof(emb_env, nullptr);
            if (emb_factor > 0.0F && emb_factor != 1.0F) {
                ops::scale_bf16(x, emb_factor, stream);
            }
        }
        nan_scan(x, "embed");
        // Control metadata and the specialized position kernel both use patch-major [P,4]
        // indexing (patch * 4 + corner); copy it without a [4,P] repack.
        copy_host(control.position_table_indices.data(), pos_indices, stream);
        copy_host(control.position_table_weights.data(), pos_weights, stream);
        if (std::getenv("NINFER_VISION_NAN") != nullptr) {
            const auto& ix = control.position_table_indices;
            const auto& wt = control.position_table_weights;
            std::int32_t lo = ix.empty() ? 0 : ix.front();
            std::int32_t hi = lo; double wsum = 0.0;
            for (const std::int32_t v : ix) { lo = std::min(lo, v); hi = std::max(hi, v); }
            for (const float v : wt) { wsum += static_cast<double>(v); }
            std::fprintf(stderr, "[vision-pos] idx=%d..%d table=%d weights=%zu wsum=%.3f\n", lo, hi,
                         static_cast<int>(config_.num_position_embeddings), wt.size(), wsum);
            std::fflush(stderr);
        }
        Tensor position_table = parameters_.position_embedding.reshape(
            {dimension(config_.hidden_size), dimension(config_.num_position_embeddings)});
        if (std::getenv("NINFER_VISION_TRACE") != nullptr) {
            (void)cudaStreamSynchronize(stream);
            const std::size_t total0 = static_cast<std::size_t>(x.ne[0]) * static_cast<std::size_t>(x.ne[1]);
            std::vector<std::uint16_t> all0(total0);
            (void)cudaMemcpy(all0.data(), x.data, total0 * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
            double s0 = 0.0, q0 = 0.0;
            for (std::size_t i = 0; i < total0; ++i) { std::uint32_t w = static_cast<std::uint32_t>(all0[i]) << 16; float v = 0.0F; std::memcpy(&v, &w, sizeof(float)); s0 += v; q0 += static_cast<double>(v) * static_cast<double>(v); }
            const double m0 = s0 / static_cast<double>(total0);
            std::fprintf(stderr, "[vision-patchonly] mean=%.4f std=%.4f\n", m0, std::sqrt(q0 / static_cast<double>(total0) - m0 * m0)); std::fflush(stderr);
        }
        ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
        nan_scan(x, "pos");
        if (std::getenv("NINFER_VISION_TRACE") != nullptr) {
            (void)cudaStreamSynchronize(stream);
            std::uint16_t probe[64] = {};
            (void)cudaMemcpy(probe, x.data, sizeof(probe), cudaMemcpyDeviceToHost);
            double sum     = 0.0;
            double squares = 0.0;
            for (std::size_t index = 0; index < 64; ++index) {
                std::uint32_t wide = static_cast<std::uint32_t>(probe[index]) << 16;
                float value        = 0.0F;
                std::memcpy(&value, &wide, sizeof(float));
                sum += static_cast<double>(value);
                squares += static_cast<double>(value) * static_cast<double>(value);
            }
            const double mean      = sum / 64.0;
            const double deviation = std::sqrt(std::max(0.0, squares / 64.0 - mean * mean));
            std::fprintf(stderr, "[vision-embed] patches=%d hidden=%d std=%.5f mean=%.5f layers=%zu\n",
                         x.ne[1], x.ne[0], deviation, mean, parameters_.layers.size());
            std::fflush(stderr);
        }
    }
    const auto check_nan = [&](const Tensor& value, std::size_t current_layer, const char* tag) {
        if (std::getenv("NINFER_VISION_NAN") == nullptr) { return; }
        (void)cudaStreamSynchronize(stream);
        const std::size_t count = static_cast<std::size_t>(value.ne[0]) * static_cast<std::size_t>(value.ne[1]);
        std::vector<std::uint16_t> host(count);
        (void)cudaMemcpy(host.data(), value.data, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        std::size_t bad = 0; double total = 0.0; std::size_t first_bad = count;
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t w = static_cast<std::uint32_t>(host[i]) << 16; float f = 0.0F;
            std::memcpy(&f, &w, sizeof(float));
            if (!std::isfinite(f)) { if (bad == 0) { first_bad = i; } ++bad; } else { total += static_cast<double>(f); }
        }
        std::fprintf(stderr, "[vision-nan] layer=%zu %s nonfinite=%zu/%zu first_token=%zu mean=%.4f\n", current_layer, tag, bad, count,
                 first_bad / static_cast<std::size_t>(config_.hidden_size), bad == count ? 0.0 : total / static_cast<double>(count - bad));
        std::fflush(stderr);
    };
    const char* dump_dir = std::getenv("NINFER_VISION_DUMP");
    auto dump_bf16 = [&](const char* tag, const Tensor& tensor) {
        if (dump_dir == nullptr) { return; }
        std::size_t count = 1;
        for (int axis = 0; axis < 4; ++axis) { count *= static_cast<std::size_t>(tensor.ne[axis] > 0 ? tensor.ne[axis] : 1); }
        if (count == 0) { return; }
        (void)cudaStreamSynchronize(stream);
        std::vector<std::uint16_t> raw(count);
        (void)cudaMemcpy(raw.data(), tensor.data, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        std::vector<float> host(count);
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(raw[i]) << 16;
            std::memcpy(&host[i], &bits, sizeof(float));
        }
        const std::string path = std::string(dump_dir) + "\\" + tag + ".f32";
        if (FILE* fp = std::fopen(path.c_str(), "wb"); fp != nullptr) {
            (void)std::fwrite(host.data(), sizeof(float), count, fp);
            std::fclose(fp);
        }
    };
    if (dump_dir != nullptr) { dump_bf16("emb_pre", x); }
    for (std::size_t layer = 0; layer < parameters_.layers.size(); ++layer) {
        nvtx::ScopedRange layer_range(nvtx::Name::VisionLayer, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(layer));
        const auto& block = parameters_.layers[layer];
        {
            nvtx::ScopedRange attention_range(nvtx::Name::VisionAttention,
                                              nvtx::Category::Attention,
                                              static_cast<std::uint64_t>(layer));
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    // No delta offset on the norm scales: the artifacts and the mmproj export both store
                    // the final gamma. The old NINFER_VISION_NORM_PLUS_ONE experiment added 1.0 here and
                    // greyed every image out, so it was removed rather than left as a trap.
                    ops::layer_norm(x, block.norm1.weight, block.norm1.bias, 1.0e-6F, h, stream);
                    if (layer == 0) { dump_bf16("l0_norm1", h); }
                    project(h, block.qkv, qkv, layout.qkv_scratch);
                }
                ops::add_bias(block.qkv_bias, qkv, stream);
                if (layer == 0) { dump_bf16("l0_qkv", qkv); }
                const std::int32_t plane      = dimension(config_.hidden_size);
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16,
                         {dimension(config_.hidden_size / config_.num_heads),
                          dimension(config_.num_heads), patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {dimension(config_.hidden_size / config_.num_heads),
                          dimension(config_.num_heads), patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {dimension(config_.hidden_size / config_.num_heads),
                          dimension(config_.num_heads), patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                ops::rope(position_ids, dimension(config_.hidden_size / config_.num_heads),
                          10'000.0F, q, k, stream);
                Tensor attended_heads =
                    attended.view({dimension(config_.hidden_size / config_.num_heads),
                                   dimension(config_.num_heads), patches});
                if (std::getenv("NINFER_VISION_TRACE") != nullptr && layer < 2) {
                    (void)cudaStreamSynchronize(stream);
                    for (int which = 0; which < 2; ++which) {
                        const Tensor& src = which == 0 ? q : k;
                        const std::size_t cnt = static_cast<std::size_t>(src.ne[0]) * static_cast<std::size_t>(src.ne[1]);
                        std::vector<std::uint16_t> hq(cnt);
                        (void)cudaMemcpy(hq.data(), src.data, cnt * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                        double sm = 0.0, sq = 0.0;
                        for (std::size_t i = 0; i < cnt; ++i) { std::uint32_t w = static_cast<std::uint32_t>(hq[i]) << 16; float f = 0.0F; std::memcpy(&f, &w, sizeof(float)); sm += f; sq += static_cast<double>(f) * static_cast<double>(f); }
                        const double mm = sm / static_cast<double>(cnt);
                        std::fprintf(stderr, "[vision-qk] layer=%zu %s mean=%.5f std=%.5f\n", layer, which == 0 ? "q" : "k", mm, std::sqrt(sq / static_cast<double>(cnt) - mm * mm));
                    }
                    std::fflush(stderr);
                }
                ops::packed_softmax_attention(
                    q, k, v,
                    {dimension(config_.hidden_size / config_.num_heads),
                     dimension(config_.num_heads), dimension(config_.num_heads)},
                    static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.hidden_size /
                                                                           config_.num_heads))),
                    std::getenv("NINFER_VISION_SEG1") != nullptr ? 1 : control.segment_length,
                    attended_heads, stream);
            }
            if (std::getenv("NINFER_VISION_TRACE") != nullptr && (layer >= 14)) {
                (void)cudaStreamSynchronize(stream);
                const std::size_t count_a = static_cast<std::size_t>(attended.ne[0]) * static_cast<std::size_t>(attended.ne[1]);
                std::vector<std::uint16_t> host_a(count_a);
                (void)cudaMemcpy(host_a.data(), attended.data, count_a * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                double sa = 0.0, qa = 0.0;
                for (std::size_t i = 0; i < count_a; ++i) { std::uint32_t w = static_cast<std::uint32_t>(host_a[i]) << 16; float f = 0.0F; std::memcpy(&f, &w, sizeof(float)); sa += f; qa += static_cast<double>(f) * static_cast<double>(f); }
                const double ma = sa / static_cast<double>(count_a);
                std::fprintf(stderr, "[vision-attn] %zu attn=%.4f/%.4f\n", layer, ma, std::sqrt(qa / static_cast<double>(count_a) - ma * ma)); std::fflush(stderr);
            }
            Tensor projected = layout.projected.bind(backing);
            project(attended, block.output, projected, layout.projection_scratch);
            ops::add_bias(block.output_bias, projected, stream);
            if (layer == 0) { dump_bf16("l0_attn_out", projected); }
            ops::residual_add(projected, x, stream);
            if (layer == 0) { dump_bf16("l0_x_attn", x); }
            check_nan(x, layer, "attn");
        }
        {
            nvtx::ScopedRange mlp_range(nvtx::Name::VisionMlp, nvtx::Category::PostMixer,
                                        static_cast<std::uint64_t>(layer));
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                const char* fix_env  = std::getenv("NINFER_VISION_NORM_FIX");
                const bool  fix_pure = fix_env != nullptr && fix_env[0] == '1';
                if (fix_pure) {
                    // The packaged checkpoint carries inflated vision norm2 weights (they climb from
                    // ~1.9 at layer 0 to ~11.5 at layer 23 while norm1 stays at ~1.3), which pushes the
                    // residual stream to std 49 and drowns the image signal in bf16 rounding. Restoring
                    // an unscaled LayerNorm here brings the tower back to the reference behaviour.
                    static std::set<std::size_t> norm2_pure_fixed;
                    if (norm2_pure_fixed.insert(layer).second) {
                        const std::size_t wn2 = static_cast<std::size_t>(block.norm2.weight.ne[0]);
                        std::vector<std::uint16_t> hw(wn2, static_cast<std::uint16_t>(0x3F80U));
                        std::vector<std::uint16_t> hb(wn2, static_cast<std::uint16_t>(0x0000U));
                        (void)cudaMemcpy(const_cast<void*>(block.norm2.weight.data), hw.data(),
                                         wn2 * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
                        (void)cudaMemcpy(const_cast<void*>(block.norm2.bias.data), hb.data(),
                                         wn2 * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
                    }
                } else if (std::getenv("NINFER_VISION_NORM2_UNIT") != nullptr) {
                    static std::set<std::size_t> norm2_unit_fixed;
                    if (norm2_unit_fixed.insert(layer).second) {
                        const std::size_t wn2 = static_cast<std::size_t>(block.norm2.weight.ne[0]);
                        std::vector<std::uint16_t> hw(wn2), hb(wn2);
                        (void)cudaMemcpy(hw.data(), block.norm2.weight.data, wn2 * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                        (void)cudaMemcpy(hb.data(), block.norm2.bias.data, wn2 * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                        double sum_w = 0.0;
                        for (std::size_t i = 0; i < wn2; ++i) { std::uint32_t u = static_cast<std::uint32_t>(hw[i]) << 16; float f = 0.0F; std::memcpy(&f, &u, sizeof(float)); sum_w += f; }
                        const float unit_scale = static_cast<float>(wn2) / static_cast<float>(sum_w);
                        for (std::size_t i = 0; i < wn2; ++i) {
                            std::uint32_t u = static_cast<std::uint32_t>(hw[i]) << 16; float f = 0.0F; std::memcpy(&f, &u, sizeof(float));
                            f *= unit_scale; std::uint32_t o = 0; std::memcpy(&o, &f, sizeof(float)); hw[i] = static_cast<std::uint16_t>(o >> 16);
                            std::uint32_t ub = static_cast<std::uint32_t>(hb[i]) << 16; float fb = 0.0F; std::memcpy(&fb, &ub, sizeof(float));
                            fb *= unit_scale; std::uint32_t ob = 0; std::memcpy(&ob, &fb, sizeof(float)); hb[i] = static_cast<std::uint16_t>(ob >> 16);
                        }
                        (void)cudaMemcpy(const_cast<void*>(block.norm2.weight.data), hw.data(), wn2 * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
                        (void)cudaMemcpy(const_cast<void*>(block.norm2.bias.data), hb.data(), wn2 * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
                    }
                }
                ops::layer_norm(x, block.norm2.weight, block.norm2.bias, 1.0e-6F, h, stream);
                if (layer == 0) { dump_bf16("l0_norm2", h); }
                if (std::getenv("NINFER_VISION_FC1_BIAS_ZERO") != nullptr) {
                    static std::set<std::size_t> fc1_bias_zeroed;
                    if (fc1_bias_zeroed.insert(layer).second) {
                        const std::size_t wn = static_cast<std::size_t>(block.fc1_bias.ne[0]);
                        std::vector<std::uint16_t> hz(wn, static_cast<std::uint16_t>(0U));
                        (void)cudaMemcpy(const_cast<void*>(block.fc1_bias.data), hz.data(),
                                         wn * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
                    }
                }
            if (std::getenv("NINFER_VISION_TRACE") != nullptr && true && true) {
                (void)cudaStreamSynchronize(stream);
                const int wn = static_cast<int>(block.norm2.weight.ne[0]);
                std::vector<std::uint16_t> wh(static_cast<std::size_t>(wn));
                (void)cudaMemcpy(wh.data(), block.norm2.weight.data, static_cast<std::size_t>(wn) * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                const int w1n = static_cast<int>(block.norm1.weight.ne[0]);
                std::vector<std::uint16_t> w1h(static_cast<std::size_t>(w1n));
                (void)cudaMemcpy(w1h.data(), block.norm1.weight.data, static_cast<std::size_t>(w1n) * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                double w1s = 0.0, w1q = 0.0;
                for (int i = 0; i < w1n; ++i) { std::uint32_t u1 = static_cast<std::uint32_t>(w1h[i]) << 16; float f1 = 0.0F; std::memcpy(&f1, &u1, sizeof(float)); w1s += f1; w1q += static_cast<double>(f1) * static_cast<double>(f1); }
                std::fprintf(stderr, "[vision-w1] layer=%zu std=%.4f mean0=%.4f\n", layer, std::sqrt(w1q / static_cast<double>(w1n) - (w1s / static_cast<double>(w1n)) * (w1s / static_cast<double>(w1n))), w1s / static_cast<double>(w1n));
                std::fflush(stderr);
                std::fflush(stderr);
                double ws = 0.0, wq = 0.0;
                for (int i = 0; i < wn; ++i) { std::uint32_t wu = static_cast<std::uint32_t>(wh[i]) << 16; float f = 0.0F; std::memcpy(&f, &wu, sizeof(float)); ws += f; wq += static_cast<double>(f) * static_cast<double>(f); }
                const double wm = ws / static_cast<double>(wn);
                std::fprintf(stderr, "[vision-w] layer=%zu n=%d mean=%.4f std=%.4f x0=%.4f x1=%.4f x2=%.4f\n", layer, wn, wm, std::sqrt(wq / static_cast<double>(wn) - wm * wm), ([](std::uint16_t v){ std::uint32_t u = static_cast<std::uint32_t>(v) << 16; float f = 0.0F; std::memcpy(&f, &u, sizeof(float)); return f; })(wh[0]), ([](std::uint16_t v){ std::uint32_t u = static_cast<std::uint32_t>(v) << 16; float f = 0.0F; std::memcpy(&f, &u, sizeof(float)); return f; })(wh[1]), ([](std::uint16_t v){ std::uint32_t u = static_cast<std::uint32_t>(v) << 16; float f = 0.0F; std::memcpy(&f, &u, sizeof(float)); return f; })(wh[2]));
                std::vector<std::uint16_t> bh(static_cast<std::size_t>(wn));
                (void)cudaMemcpy(bh.data(), block.norm2.bias.data, static_cast<std::size_t>(wn) * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                double bs = 0.0, bq = 0.0;
                for (int i = 0; i < wn; ++i) { std::uint32_t ub = static_cast<std::uint32_t>(bh[i]) << 16; float fb = 0.0F; std::memcpy(&fb, &ub, sizeof(float)); bs += fb; bq += static_cast<double>(fb) * static_cast<double>(fb); }
                std::fprintf(stderr, "[vision-b] layer=%zu std=%.4f mean=%.4f\n", layer, std::sqrt(bq / wn - (bs / wn) * (bs / wn)), bs / wn);
                std::fflush(stderr);
                std::fflush(stderr);
            }
            if (std::getenv("NINFER_VISION_TRACE") != nullptr && (layer >= 14)) {
                (void)cudaStreamSynchronize(stream);
                const std::size_t count_h = static_cast<std::size_t>(h.ne[0]) * static_cast<std::size_t>(h.ne[1]);
                std::vector<std::uint16_t> host_h(count_h);
                (void)cudaMemcpy(host_h.data(), h.data, count_h * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                double sh = 0.0, qh = 0.0;
                for (std::size_t i = 0; i < count_h; ++i) { std::uint32_t w = static_cast<std::uint32_t>(host_h[i]) << 16; float f = 0.0F; std::memcpy(&f, &w, sizeof(float)); sh += f; qh += static_cast<double>(f) * static_cast<double>(f); }
                const double mh = sh / static_cast<double>(count_h);
                std::fprintf(stderr, "[vision-norm] %zu h=%.4f/%.4f\n", layer, mh, std::sqrt(qh / static_cast<double>(count_h) - mh * mh)); std::fflush(stderr);
            }
                project(h, block.fc1, up, layout.up_scratch);
            }
            if (std::getenv("NINFER_VISION_NO_FC1_BIAS") == nullptr) {
                ops::add_bias(block.fc1_bias, up, stream);
            }
            if (layer == 0) { dump_bf16("l0_up", up); }
            ops::gelu(up, ops::GeluMode::Tanh, stream);
            project(up, block.fc2, down, layout.down_scratch);
            ops::add_bias(block.fc2_bias, down, stream);
            if (layer == 0) { dump_bf16("l0_down", down); }
            if (std::getenv("NINFER_VISION_TRACE") != nullptr && (layer >= 14)) {
                (void)cudaStreamSynchronize(stream);
                const std::size_t count_d = static_cast<std::size_t>(down.ne[0]) * static_cast<std::size_t>(down.ne[1]);
                std::vector<std::uint16_t> host_d(count_d);
                (void)cudaStreamSynchronize(stream);
                const std::size_t count_u = static_cast<std::size_t>(up.ne[0]) * static_cast<std::size_t>(up.ne[1]);
                std::vector<std::uint16_t> host_u(count_u);
                (void)cudaMemcpy(host_u.data(), up.data, count_u * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                double su = 0.0, qu = 0.0;
                for (std::size_t i = 0; i < count_u; ++i) { std::uint32_t wu = static_cast<std::uint32_t>(host_u[i]) << 16; float fu = 0.0F; std::memcpy(&fu, &wu, sizeof(float)); su += fu; qu += static_cast<double>(fu) * static_cast<double>(fu); }
                const double mu = su / static_cast<double>(count_u);
                (void)cudaMemcpy(host_d.data(), down.data, count_d * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                double sd = 0.0, qd = 0.0;
                for (std::size_t i = 0; i < count_d; ++i) { std::uint32_t w = static_cast<std::uint32_t>(host_d[i]) << 16; float f = 0.0F; std::memcpy(&f, &w, sizeof(float)); sd += f; qd += static_cast<double>(f) * static_cast<double>(f); }
                const double md = sd / static_cast<double>(count_d);
                std::fprintf(stderr, "[vision-mlp] %zu up=%.3f/%.3f down=%.4f/%.4f\n", layer, mu, std::sqrt(qu / static_cast<double>(count_u) - mu * mu), md, std::sqrt(qd / static_cast<double>(count_d) - md * md)); std::fflush(stderr);
            }
            ops::residual_add(down, x, stream);
            check_nan(x, layer, "mlp");
        }
        if (std::getenv("NINFER_VISION_TRACE") != nullptr) {
            (void)cudaStreamSynchronize(stream);
            std::uint16_t probe[4] = {};
            const std::size_t step = static_cast<std::size_t>(x.ne[0]);
            for (std::size_t t = 0; t < 4; ++t) { (void)cudaMemcpy(probe + t, static_cast<const std::uint8_t*>(x.data) + t * step * 2U, sizeof(std::uint16_t), cudaMemcpyDeviceToHost); }
            std::fprintf(stderr, "[vision-layer] %zu:", layer);
            for (std::size_t t = 0; t < 4; ++t) { std::uint32_t w = static_cast<std::uint32_t>(probe[t]) << 16; float v = 0.0F; std::memcpy(&v, &w, sizeof(float)); std::fprintf(stderr, " %.4f", v); }
            std::fprintf(stderr, "\n"); std::fflush(stderr);
            const std::size_t total = static_cast<std::size_t>(x.ne[0]) * static_cast<std::size_t>(x.ne[1]);
            std::vector<std::uint16_t> all(total);
            (void)cudaMemcpy(all.data(), x.data, total * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
            double s = 0.0, q = 0.0;
            for (std::size_t i = 0; i < total; ++i) { std::uint32_t w = static_cast<std::uint32_t>(all[i]) << 16; float v = 0.0F; std::memcpy(&v, &w, sizeof(float)); s += v; q += static_cast<double>(v) * static_cast<double>(v); }
            const double m = s / static_cast<double>(total);
            std::fprintf(stderr, "[vision-stat] %zu mean=%.4f std=%.4f\n", layer, m, std::sqrt(q / static_cast<double>(total) - m * m)); std::fflush(stderr);
            std::uint16_t weights[2] = {};
            (void)cudaMemcpy(weights, block.norm1.weight.data, sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(weights + 1, block.qkv_bias.data, sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
            std::uint32_t a = static_cast<std::uint32_t>(weights[0]) << 16, b = static_cast<std::uint32_t>(weights[1]) << 16;
            float w0 = 0.0F, b0 = 0.0F; std::memcpy(&w0, &a, sizeof(float)); std::memcpy(&b0, &b, sizeof(float));
            std::fprintf(stderr, "[vision-weight] %zu norm1=%.4f qkvbias=%.6f\n", layer, w0, b0); std::fflush(stderr);
        }
    }

    {
        nvtx::ScopedRange merge_range(nvtx::Name::VisionMerge, nvtx::Category::Vision,
                                      static_cast<std::uint64_t>(tokens64));
        Tensor normalized = layout.normalized.bind(backing);
        dump_bf16("tower_out", x);
        if (std::getenv("NINFER_VISION_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[vision-merge] x ne=%lld,%lld nb=%lld,%lld norm ne=%lld,%lld nb=%lld,%lld "
                         "tokens=%zu\n",
                         static_cast<long long>(x.ne[0]), static_cast<long long>(x.ne[1]),
                         static_cast<long long>(x.nb[0]), static_cast<long long>(x.nb[1]),
                         static_cast<long long>(normalized.ne[0]),
                         static_cast<long long>(normalized.ne[1]),
                         static_cast<long long>(normalized.nb[0]),
                         static_cast<long long>(normalized.nb[1]), tokens);
            std::fflush(stderr);
        }
        if (const char* mn_env = std::getenv("NINFER_VISION_MERGER_NORM_UNIT"); mn_env != nullptr) {
            static bool merger_norm_done = false;
            if (!merger_norm_done) {
                merger_norm_done = true;
                const std::size_t wn = static_cast<std::size_t>(parameters_.merger_norm.weight.ne[0]);
                std::vector<std::uint16_t> hw(wn, static_cast<std::uint16_t>(0x3F80U));
                std::vector<std::uint16_t> hb(wn, static_cast<std::uint16_t>(0x0000U));
                (void)cudaMemcpy(const_cast<void*>(parameters_.merger_norm.weight.data), hw.data(),
                                 wn * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
                (void)cudaMemcpy(const_cast<void*>(parameters_.merger_norm.bias.data), hb.data(),
                                 wn * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
            }
        }
        const bool use_post_ln = g_post_ln_weight != nullptr && g_post_ln_bias != nullptr &&
                                 std::getenv("NINFER_VISION_POST_LN") != nullptr;
        if (use_post_ln) {
            Tensor pw(g_post_ln_weight, DType::BF16,
                      {static_cast<std::int32_t>(config_.hidden_size)});
            Tensor pb(g_post_ln_bias, DType::BF16,
                      {static_cast<std::int32_t>(config_.hidden_size)});
            ops::layer_norm(x, pw, pb, 1.0e-6F, normalized, stream);
            ops::layer_norm(normalized, parameters_.merger_norm.weight, parameters_.merger_norm.bias,
                            1.0e-6F, x, stream);
        } else {
            ops::layer_norm(x, parameters_.merger_norm.weight, parameters_.merger_norm.bias, 1.0e-6F,
                            normalized, stream);
        }
        Tensor merged = (use_post_ln ? x : normalized)
                           .view({dimension(config_.merger_width()), tokens});
        Tensor hidden = layout.merger_hidden.bind(backing);
        project(merged, parameters_.merger_fc1, hidden, layout.merger_first_scratch);
        ops::add_bias(parameters_.merger_fc1_bias, hidden, stream);
        ops::gelu(hidden, ops::GeluMode::Exact, stream);
        project(hidden, parameters_.merger_fc2, output, layout.merger_second_scratch);
        ops::add_bias(parameters_.merger_fc2_bias, output, stream);
        dump_bf16("merged_out", output);
    // The merger is trained to emit its features in the text model's own embedding space, so the
    // tokens are spliced at full amplitude.  The former 0.1 default was a workaround that kept the
    // colours readable but left the content too weak for the model to localise: each single-square
    // query collapsed onto the same text prior regardless of where the square actually was.
    // NINFER_VISION_SCALE remains available for experiments only.
    if (const char* scale_env = std::getenv("NINFER_VISION_SCALE"); scale_env != nullptr) {
        const float factor = std::strtof(scale_env, nullptr);
        if (factor > 0.0F && factor != 1.0F) { ops::scale_bf16(output, factor, stream); }
    }
    }
    if (std::getenv("NINFER_VISION_TRACE") != nullptr) {
        // A constant embedding here means the vision tower never sees the patch colours; a varying
        // one means the fault is downstream in how the tokens reach the text model.
        (void)cudaStreamSynchronize(stream);
        std::uint16_t host[8] = {};
        double sum            = 0.0;
        double squares        = 0.0;
        const std::size_t samples = static_cast<std::size_t>(std::min(output.ne[0] * output.ne[1], 8));
        (void)cudaStreamSynchronize(stream);
        (void)cudaMemcpy(host, output.data, samples * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "[vision-out] hidden=%d tokens=%d first=", output.ne[0], output.ne[1]);        for (std::size_t index = 0; index < samples; ++index) {
            std::uint32_t wide = static_cast<std::uint32_t>(host[index]) << 16;
            float value        = 0.0F;
            std::memcpy(&value, &wide, sizeof(float));
            sum += static_cast<double>(value);
            squares += static_cast<double>(value) * static_cast<double>(value);
            std::fprintf(stderr, "%.4f ", value);
        }
        if (samples != 0) {
            const double mean      = sum / static_cast<double>(samples);
            const double deviation = std::sqrt(std::max(0.0, squares / static_cast<double>(samples) -
                                                                 mean * mean));
            std::fprintf(stderr, "mean=%.4f std=%.4f", mean, deviation);
        }
        std::fprintf(stderr, "\n");
        // Same feature dimension sampled across tokens: if these collapse, the tower is emitting one
        // vector for every position and the image cannot be described.
        const std::size_t stride = static_cast<std::size_t>(output.ne[0]);
        {
            const std::size_t tsel[2] = {0U, 90U}, dsel[5] = {0U, 100U, 1000U, 3000U, 5000U};
            for (std::size_t ti = 0; ti < 2; ++ti) {
                std::fprintf(stderr, "[vision-dim] token=%zu:", tsel[ti]);
                for (std::size_t di = 0; di < 5; ++di) {
                    std::uint16_t raw = 0;
                    (void)cudaMemcpy(&raw, static_cast<const std::uint8_t*>(output.data) + (tsel[ti] * stride + dsel[di]) * 2U, sizeof(raw), cudaMemcpyDeviceToHost);
                    std::uint32_t wide = static_cast<std::uint32_t>(raw) << 16; float value = 0.0F; std::memcpy(&value, &wide, sizeof(float));
                    std::fprintf(stderr, " d%zu=%.4f", dsel[di], value);
                }
                std::fprintf(stderr, "\n");
            }
            std::fflush(stderr);
        }
        std::uint16_t across[6]  = {};
        for (std::size_t token = 0; token < 6 && token * stride < static_cast<std::size_t>(
                                                                     output.ne[0] * output.ne[1]);
             ++token) {
            (void)cudaMemcpy(across + token, static_cast<const std::uint8_t*>(output.data) + token * stride * 2U,
                             sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        }
        std::fprintf(stderr, "[vision-across] ");
        for (std::size_t token = 0; token < 6; ++token) {
            std::uint32_t wide = static_cast<std::uint32_t>(across[token]) << 16;
            float value        = 0.0F;
            std::memcpy(&value, &wide, sizeof(float));
            std::fprintf(stderr, "%.4f ", value);
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
}

VisionPrefillSession::VisionPrefillSession(
    DeviceContext& device, const execution::Parameters& parameters, DeviceSpan workspace,
    const VisionWorkspacePlan& workspace_plan, qwen3_5::PreparedPromptData& prompt,
    const VisionPrefillPlan& plan, std::size_t& handoff_peak_bytes)
    : device_(device), workspace_(workspace), workspace_plan_(workspace_plan), prompt_(prompt),
      plan_(plan), handoff_peak_bytes_(handoff_peak_bytes), context_(device, parameters) {
    if (plan_.control == nullptr || plan_.control->items.empty() || plan_.uses.empty()) {
        throw std::invalid_argument("Vision prefill plan has no suffix item spans");
    }
    if (workspace_.data == nullptr || workspace_.bytes < workspace_plan_.capacity_bytes ||
        plan_.max_merged_count == 0 || plan_.max_merged_count > workspace_plan_.max_merged_tokens) {
        throw std::invalid_argument("Vision prefill workspace plan is invalid");
    }
    std::uint32_t previous_end = 0;
    std::optional<std::uint32_t> previous_item;
    for (const VisionUseSpan& use : plan_.uses) {
        if (use.begin >= use.end || use.begin < previous_end ||
            use.end > prompt_.token_ids.size()) {
            throw std::invalid_argument("Vision suffix item spans are invalid or unordered");
        }
        if (use.control_index >= plan_.control->items.size() ||
            use.prepared_item_index >= prompt_.vision_items.size() ||
            use.prepared_item_index >= prompt_.media_payloads.size() ||
            plan_.control->prepared_item_begin + use.control_index != use.prepared_item_index ||
            (previous_item && use.prepared_item_index <= *previous_item)) {
            throw std::invalid_argument("Vision suffix item indices are invalid or unordered");
        }
        const qwen3_5::VisionItemControl& control = plan_.control->items[use.control_index];
        const qwen3_5::VisionItem& source         = prompt_.vision_items[use.prepared_item_index];
        if (control.scatter_indices.empty() ||
            use.end != static_cast<std::uint32_t>(control.scatter_indices.back()) + 1U ||
            (use.begin != static_cast<std::uint32_t>(control.scatter_indices.front()) &&
             use.begin + 1U != static_cast<std::uint32_t>(control.scatter_indices.front())) ||
            source.modality != control.modality || source.grid.temporal != control.grid.temporal ||
            source.grid.height != control.grid.height || source.grid.width != control.grid.width ||
            source.patch_begin != control.patch_begin ||
            source.patch_count != control.patch_count) {
            throw std::invalid_argument("Vision suffix plan does not describe the prepared item");
        }
        if (control.merged_count > plan_.max_merged_count) {
            throw std::invalid_argument("Vision suffix item exceeds its request workspace extent");
        }
        const Tensor output =
            VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);
        const std::size_t patch_elements = checked_mul(
            control.patch_count, static_cast<std::size_t>(context_.config().patch_width()),
            "item patch elements");
        const auto& payload = prompt_.media_payloads[use.prepared_item_index];
        if (output.bytes() > workspace_plan_.handoff_capacity_bytes || !payload ||
            payload->patch_elements != patch_elements) {
            throw std::invalid_argument("Vision suffix item storage has an invalid shape");
        }
        previous_end  = use.end;
        previous_item = use.prepared_item_index;
    }
    if (plan_.max_merged_count != 0 &&
        std::none_of(plan_.control->items.begin(), plan_.control->items.end(),
                     [&](const qwen3_5::VisionItemControl& item) {
                         return item.merged_count == plan_.max_merged_count;
                     })) {
        throw std::invalid_argument("Vision request workspace extent has no matching suffix item");
    }
    encoded_payloads_pending_release_.reserve(plan_.uses.size());
    timers_.reserve(plan_.uses.size());
}

VisionChunk VisionPrefillSession::prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length) {
    if (nominal_length == 0 || begin >= prompt_.token_ids.size()) {
        throw std::invalid_argument("Vision chunk range is empty or outside the prompt");
    }
    const std::uint64_t nominal_end64 =
        static_cast<std::uint64_t>(begin) + static_cast<std::uint64_t>(nominal_length);
    std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(nominal_end64, prompt_.token_ids.size()));

    while (next_use_ < plan_.uses.size() && plan_.uses[next_use_].end <= begin) { ++next_use_; }
    const VisionUseSpan* active = nullptr;
    if (next_use_ < plan_.uses.size() && plan_.uses[next_use_].begin < end) {
        active = &plan_.uses[next_use_];
        if (next_use_ + 1U < plan_.uses.size()) {
            end = std::min(end, plan_.uses[next_use_ + 1U].begin);
        }
    }
    if (end <= begin) { throw std::logic_error("Vision chunk cap made no forward progress"); }
    if (active == nullptr) {
        return VisionChunk{static_cast<std::int32_t>(end - begin), nullptr, {}};
    }
    const qwen3_5::VisionItemControl& control = plan_.control->items[active->control_index];
    Tensor output = VisionContext::bind_output(workspace_, workspace_plan_, control.merged_count);

    if (!active_item_ || *active_item_ != active->prepared_item_index) {
        const auto& payload = prompt_.media_payloads[active->prepared_item_index];
        timers_.emplace_back(device_);
        timers_.back().start();
        context_.encode(VisionItemView{payload->span(), &control}, output, workspace_,
                        workspace_plan_);
        timers_.back().record_stop();
        active_item_          = active->prepared_item_index;
        active_handoff_bytes_ = output.bytes();
        handoff_peak_bytes_   = std::max(handoff_peak_bytes_, active_handoff_bytes_);
        encoded_payloads_pending_release_.push_back(active->prepared_item_index);
    }
    return VisionChunk{static_cast<std::int32_t>(end - begin), &control, output};
}

void VisionPrefillSession::release_encoded_media_payloads() noexcept {
    for (const std::uint32_t item_index : encoded_payloads_pending_release_) {
        if (item_index >= prompt_.media_payloads.size()) { std::terminate(); }
        prompt_.media_payloads[item_index].reset();
    }
    encoded_payloads_pending_release_.clear();
}

void VisionPrefillSession::retire_handoff() noexcept {
    active_item_.reset();
    active_handoff_bytes_ = 0;
}

double VisionPrefillSession::elapsed_seconds() const {
    double milliseconds = 0.0;
    for (const CudaEventTimer& timer : timers_) { milliseconds += timer.elapsed_ms(); }
    return milliseconds / 1000.0;
}

} // namespace ninfer::models::qwen3_5::execution
