#include "models/qwen3_5/load.h"

#include "artifact/views.h"
#include "core/weight_view.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <iostream>
#include <cstring>
#include <memory>
#include <utility>

namespace ninfer::models::qwen3_5 {

namespace {

// iamwavecut v3: per-use hadamard_signs auxiliary. Returns true when any Use carries one.
bool has_v3_hadamard_signs(const artifact::Directory& directory) {
    for (const auto& [key, use] : directory.uses) {
        if (use.auxiliaries.contains("hadamard_signs")) { return true; }
    }
    return false;
}

// bf16 -> fp32: bf16 is the upper half of an fp32 word (1+8+7 bits), so widening is a 16-bit
// left shift. Works for any payload including the ±1 sign rows of the folded basis.
float bf16_to_fp32(std::uint16_t bits) {
    std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float out = 0.0f;
    static_assert(sizeof(out) == sizeof(word));
    std::memcpy(&out, &word, sizeof(out));
    return out;
}

} // namespace

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    // AMD port (shensanshu ternary): folded-basis sign block, present on ternary artifacts.
    std::unique_ptr<FoldedSignTable> signs;
    artifact::ParameterReference signs_reference;
    // AMD port (iamwavecut v3): per-use hadamard_signs auxiliary objects, bf16, one per width.
    // We read them into host memory during planning and upload a converted FP32 table here.
    std::vector<std::byte> v3_signs_host;      // 28672 * 4 bytes, FP32, concatenated by width
    std::vector<std::pair<std::int32_t, std::uint64_t>> v3_sign_widths;  // (width, offset)
    std::unique_ptr<DeviceBuffer> extra_device;  // owns the uploaded v3 sign table
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    // AMD port: optional folded-basis sign block (ternary artifacts carry text/hadamard_signs).
    if (binder.contains("text/hadamard_signs")) {
        auto signs_ref  = binder.parameter("text/hadamard_signs", {28672},
                                          artifact::Residency::Device, QType::FP32);
        auto widths_ref = binder.parameter("text/hadamard_widths", {3}, artifact::Residency::Host,
                                           QType::INT32);
        const auto widths = binder.values(widths_ref.binding, QType::INT32).integers();
        auto table        = std::make_unique<FoldedSignTable>();
        std::uint64_t offset = 0;
        for (const auto width : widths) {
            if (width <= 0 || width % 1024) {
                throw artifact::ArtifactError(
                    "hadamard widths must be positive multiples of 1024");
            }
            table->width_offsets.push_back({static_cast<std::int32_t>(width), offset});
            offset += static_cast<std::uint64_t>(width);
        }
        if (offset != 28672) {
            throw artifact::ArtifactError("hadamard widths must partition the 28672 sign rows");
        }
        out->signs           = std::move(table);
        out->signs_reference = std::move(signs_ref);
    } else if (has_v3_hadamard_signs(reader.directory())) {
        // iamwavecut v3: per-use hadamard_signs auxiliary objects (bf16, one per width).
        // Read them, convert to FP32 and concatenate into the same 28672-row table.
        const auto& directory = reader.directory();
        struct SignEntry { std::int32_t width; artifact::ObjectHandle object; };
        std::vector<SignEntry> entries;
        std::vector<artifact::ObjectHandle> seen;
        for (const auto& [key, use] : directory.uses) {
            const auto found = use.auxiliaries.find("hadamard_signs");
            if (found == use.auxiliaries.end()) { continue; }
            for (const auto& part : found->second.parts) {
                if (std::find(seen.begin(), seen.end(), part.object) != seen.end()) { continue; }
                seen.push_back(part.object);
                const auto& geometry = reader.geometry(part.object);
                if (geometry.shape.size() != 1 || geometry.shape[0] == 0 ||
                    geometry.shape[0] % 1024 != 0) {
                    throw artifact::ArtifactError(
                        "v3 hadamard_signs auxiliary must be a [K] row with K % 1024 == 0");
                }
                entries.push_back(
                    {static_cast<std::int32_t>(geometry.shape[0]), part.object});
            }
        }
        if (entries.empty()) {
            throw artifact::ArtifactError("v3 hadamard_signs: no auxiliary rows");
        }
        std::sort(entries.begin(), entries.end(),
                  [](const SignEntry& a, const SignEntry& b) { return a.width < b.width; });
        std::uint64_t offset = 0;
        for (const auto& entry : entries) {
            out->v3_sign_widths.push_back({entry.width, offset});
            offset += static_cast<std::uint64_t>(entry.width);
        }
        if (offset != 28672) {
            throw artifact::ArtifactError("v3 hadamard_signs widths must sum to 28672 (got " +
                                          std::to_string(offset) + ")");
        }
        out->v3_signs_host.reserve(static_cast<std::size_t>(28672) * 4);
        for (const auto& entry : entries) {
            const auto raw = reader.read_object(entry.object);
            if (raw.size() != static_cast<std::size_t>(entry.width) * 2) {
                throw artifact::ArtifactError("v3 hadamard_signs: bf16 byte count mismatch");
            }
            for (std::int32_t i = 0; i < entry.width; ++i) {
                std::uint16_t bits = static_cast<std::uint16_t>(
                    std::to_integer<std::uint16_t>(raw[2 * i]) |
                    (static_cast<std::uint16_t>(std::to_integer<std::uint16_t>(raw[2 * i + 1]))
                     << 8));
                const float value = bf16_to_fp32(bits);
                const auto* p     = reinterpret_cast<const std::byte*>(&value);
                out->v3_signs_host.insert(out->v3_signs_host.end(), p, p + sizeof(float));
            }
        }
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish();
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    std::cerr << "[mini] mat-resolve\n";
    std::cerr.flush();
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    // AMD port: the sign table's base lands in device memory during materialization.
    if (data->signs) {
        auto view = artifact::bind_view(data->signs_reference, backing);
        data->signs->base =
            reinterpret_cast<const float*>(contiguous_weight_region(view).parent->data);
    } else if (!data->v3_signs_host.empty()) {
        // iamwavecut v3: upload the converted FP32 table into a dedicated device buffer.
        auto buffer = std::make_unique<DeviceBuffer>(data->v3_signs_host.size());
        buffer->copy_from_host(data->v3_signs_host.data(), data->v3_signs_host.size());
        auto table = std::make_unique<FoldedSignTable>();
        table->base           = reinterpret_cast<const float*>(buffer->p);
        table->width_offsets  = std::move(data->v3_sign_widths);
        data->signs           = std::move(table);
        data->extra_device    = std::move(buffer);
    }
    std::cerr << "[mini] mat-model-ctor\n";
    std::cerr.flush();
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(backing),
        std::move(data->signs), std::move(data->extra_device)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace ninfer::models::qwen3_5
