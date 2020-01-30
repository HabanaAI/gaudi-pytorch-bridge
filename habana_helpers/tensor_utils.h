#pragma once

#include <synapse_helpers/habana_tensor.h>
#include <torch/script.h>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace habana_helpers {
c10::ScalarType scalar_type(const c10::Scalar& s);

at::Tensor scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::TensorOptions& options,
    unsigned num_dimensions);

synapse_helpers::tensor create_tensor(
    const at::Tensor& tensor,
    const synGraphHandle graph,
    bool persistent,
    const c10::optional<c10::ScalarType> dtype = c10::nullopt);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<const at::Tensor*> tensors,
    synGraphHandle graph,
    const std::vector<bool> persistents,
    const std::vector<c10::optional<c10::ScalarType>> dtypes);

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
create_tensors(
    const std::vector<const at::Tensor*> tensors,
    synGraphHandle graph,
    bool persistent);

synapse_helpers::tensor duplicate_tensor_in_memory_section(
    const synapse_helpers::tensor& tensor);

std::vector<void*> extract_data_ptrs(const std::vector<const at::Tensor*>& vec);

std::vector<std::string> names(const std::vector<synapse_helpers::tensor>&);

std::string name_suffix_from_type(const c10::ScalarType pt_type);

at::Tensor to_cpu(const at::Tensor& hpu_tensor);
} // namespace habana_helpers
