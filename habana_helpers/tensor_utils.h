#pragma once

#include <synapse_helpers/habana_tensor.h>
#include <torch/script.h>
#include <unordered_map>

namespace habana_helpers {
synDataType pytorch_to_synapse_type(c10::ScalarType pt_type);

synapse_helpers::tensor create_tensor(
    const at::Tensor& t,
    std::string name,
    bool persistent);
} // namespace habana_helpers
