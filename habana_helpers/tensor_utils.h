#pragma once

#include <synapse_helpers/habana_tensor.h>
#include <torch/script.h>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace habana_helpers {
synDataType pytorch_to_synapse_type(const c10::ScalarType pt_type);
synDataType pytorch_to_synapse_type(const c10::Scalar& s);
c10::ScalarType scalar_type(const c10::Scalar& s);

synapse_helpers::tensor create_tensor(
    const at::Tensor& t,
    const std::string& name,
    const synGraphHandle graph,
    bool persistent);

synapse_helpers::tensor create_tensor(
    const at::Tensor& t,
    const std::string& name,
    const synGraphHandle graph,
    bool persistent,
    const c10::ScalarType dtype);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<const at::Tensor*>& tensors,
    const std::vector<std::string>& names,
    const synGraphHandle graph,
    const std::vector<bool>& persistents);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<const at::Tensor*>& tensors,
    const std::vector<std::string>& names,
    const synGraphHandle graph,
    const std::vector<bool>& persistents,
    const std::vector<c10::optional<c10::ScalarType>>& dtypes);

at::Tensor scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::TensorOptions& options,
    const unsigned num_dimensions);

std::vector<std::string> names(const std::vector<synapse_helpers::tensor>&);

std::string name_suffix_from_type(const c10::ScalarType pt_type);

at::Tensor to_cpu(const at::Tensor& hpu_tensor);
} // namespace habana_helpers
