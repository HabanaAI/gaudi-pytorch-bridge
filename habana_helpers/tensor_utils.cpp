#include <synapse_helpers/graph.h>
#include <algorithm>

#include "habana_device/HPUCheck.h"
#include "habana_device/fake_tensor_builder.h"
#include "habana_device/hpu_cached_devices.h"
#include "tensor_utils.h"

at::Tensor habana_helpers::to_cpu(const at::Tensor& hpu_tensor) {
  if (hpu_tensor.defined())
    return hpu_tensor.to(at::DeviceType::CPU);
  else
    return hpu_tensor;
}

synDataType habana_helpers::pytorch_to_synapse_type(
    const c10::ScalarType pt_type) {
  static const std::unordered_map<c10::ScalarType, synDataType> map{
      {c10::ScalarType::Byte, synDataType::syn_type_uint8},
      {c10::ScalarType::Char, synDataType::syn_type_int8},
      {c10::ScalarType::Short, synDataType::syn_type_int16},
      {c10::ScalarType::Int, synDataType::syn_type_int32},
      //   {c10::ScalarType::Long , synDataType::},
      {c10::ScalarType::Float, synDataType::syn_type_float},
      //   {c10::ScalarType::Double , synDataType::},
      //   {c10::ScalarType::Bool , synDataType::},
      {c10::ScalarType::BFloat16, synDataType::syn_type_bf16},
  };

  auto result = map.find(pt_type);
  TORCH_CHECK(result != map.end(), "Unsupported pytorch type ", pt_type);

  return result->second;
}

synDataType habana_helpers::pytorch_to_synapse_type(const c10::Scalar& s) {
  return habana_helpers::pytorch_to_synapse_type(
      habana_helpers::scalar_type(s));
}

c10::ScalarType habana_helpers::scalar_type(const c10::Scalar& s) {
  if (s.isFloatingPoint()) {
    return c10::ScalarType::Float;
  } else if (s.isIntegral(false)) {
    return c10::ScalarType::Int;
  } else if (s.isBoolean()) {
    return c10::ScalarType::Bool;
  } else
    TORCH_CHECK(!s.isComplex(), "Habana doesn't support complex types");
  throw std::runtime_error("Unknown type");
}

at::Tensor habana_helpers::scalar_to_device_tensor(
    const at::Scalar& scalar,
    const at::TensorOptions& options,
    const unsigned num_dimensions) {
  TORCH_CHECK(
      scalar.isFloatingPoint(),
      "scalar_to_device_tensor currently supports only float");
  TORCH_CHECK(
      options.device().type() == c10::DeviceType::HABANA,
      "Wrong device: ",
      options.device().type());
  auto output = at::empty(std::vector<int64_t>(num_dimensions, 1), options);
  auto val = scalar.to<float>();
  synapse_helpers::HPURegistrar::get_device(options.device().index())
      .copy_data_to_device(
          &val, reinterpret_cast<synapse_helpers::device_ptr>(output.data_ptr()), output.nbytes());

  return output;
}

synapse_helpers::tensor habana_helpers::create_tensor(
    const at::Tensor& tensor,
    const std::string& name,
    const synGraphHandle graph,
    const bool persistent) {
  return habana_helpers::create_tensor(
      tensor, name, graph, persistent, tensor.scalar_type());
}

synapse_helpers::tensor habana_helpers::create_tensor(
    const at::Tensor& tensor,
    const std::string& name,
    const synGraphHandle graph,
    const bool persistent,
    const c10::ScalarType dtype) {
  auto syn_type = habana_helpers::pytorch_to_synapse_type(dtype);
  return synapse_helpers::tensor_builder::create_tensor(
      tensor.device().index(),
      syn_type,
      tensor.numel() * sizeof(syn_type),
      tensor.sizes().size(),
      tensor.sizes(),
      name,
      graph,
      persistent);
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
habana_helpers::create_tensors(
    const std::vector<const at::Tensor*>& tensors,
    const std::vector<std::string>& names,
    const synGraphHandle graph,
    const std::vector<bool>& persistents) {
  return habana_helpers::create_tensors(
      tensors,
      names,
      graph,
      persistents,
      std::vector<c10::optional<c10::ScalarType>>(
          tensors.size(), c10::nullopt));
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
habana_helpers::create_tensors(
    const std::vector<const at::Tensor*>& tensors,
    const std::vector<std::string>& names,
    const synGraphHandle graph,
    const std::vector<bool>& persistents,
    const std::vector<c10::optional<c10::ScalarType>>& dtypes) {
  const auto num_tensors = tensors.size();
  TORCH_CHECK(names.size() == num_tensors);
  TORCH_CHECK(persistents.size() == num_tensors);
  TORCH_CHECK(dtypes.size() == num_tensors);

  // tensor_helpers are used for tenor lifetime managment
  // syn_tensors are convinient to use with synapse API
  std::vector<synapse_helpers::tensor> tensor_helpers;
  std::vector<synTensor> syn_tensors;

  tensor_helpers.reserve(num_tensors);
  syn_tensors.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    tensor_helpers.push_back(habana_helpers::create_tensor(
        *tensors[i],
        names[i],
        graph,
        persistents[i],
        dtypes[i].value_or(tensors[i]->scalar_type())));
    syn_tensors.push_back(tensor_helpers[i].get());
  }

  return {std::move(tensor_helpers), std::move(syn_tensors)};
}

std::vector<std::string> habana_helpers::names(
    const std::vector<synapse_helpers::tensor>& vec) {
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(), vec.end(), std::back_inserter(names), [](auto& x) {
        return x.tensor_name_;
      });

  return names;
}

std::string habana_helpers::name_suffix_from_type(
    const c10::ScalarType pt_type) {
  auto string_or_error = synapse_helpers::graph::name_suffix_from_type(
      pytorch_to_synapse_type(pt_type));
  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          string_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(string_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  } else {
    return absl::get<std::string>(string_or_error);
  }
}
