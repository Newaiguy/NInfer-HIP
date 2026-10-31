#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing,
             std::unique_ptr<FoldedSignTable> folded_signs,
             std::unique_ptr<DeviceBuffer> extra_device)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)),
      info_(std::move(info)), folded_signs_(std::move(folded_signs)),
      extra_device_(std::move(extra_device)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    ops::WeightInput result{parameter.view, use.policy, use.activation_input_divisor,
                            folded_signs_.get()};
    if (use.input_columns) {
        const auto& columns = weight(*use.input_columns).view;
        result.input_columns =
            weight_tensor(columns, {static_cast<std::int32_t>(columns.shape[0])});
    }
    return result;
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

} // namespace ninfer::models::qwen3_5
