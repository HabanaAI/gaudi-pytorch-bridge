#pragma once

#include <synapse_helpers/habana_tensor.h>
#include <torch/script.h>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace habana_helpers {
synDataType pytorch_to_synapse_type(const c10::ScalarType pt_type);

synapse_helpers::tensor create_tensor(
    const at::Tensor& t,
    std::string name,
    bool persistent);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<const at::Tensor*> tensors,
    const std::vector<std::string> names,
    const std::vector<bool> persistents);

std::vector<std::string> names(const std::vector<synapse_helpers::tensor>&);

std::string name_suffix_from_type(const c10::ScalarType pt_type);
} // namespace habana_helpers
