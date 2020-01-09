#include <algorithm>

#include "tensor_utils.h"
#include "habana_device/fake_tensor_builder.h"

synDataType habana_helpers::pytorch_to_synapse_type(c10::ScalarType pt_type) {
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

synapse_helpers::tensor habana_helpers::create_tensor(
    const at::Tensor& tensor,
    std::string name,
    bool persistent) {
  return synapse_helpers::tensor_builder::create_tensor(
      tensor.device().index(),
      pytorch_to_synapse_type(tensor.scalar_type()),
      tensor.nbytes(),
      tensor.sizes().size(),
      tensor.sizes(),
      name,
      persistent);
}

std::tuple<std::vector<synapse_helpers::tensor>, std::vector<synTensor>>
habana_helpers::create_tensors(
    const std::vector<const at::Tensor*> tensors,
    const std::vector<std::string> names,
    const std::vector<bool> persistents) {
  const auto num_tensors = tensors.size();
  TORCH_CHECK(names.size() == num_tensors);
  TORCH_CHECK(persistents.size() == num_tensors);

  // tensor_helpers are used for tenor lifetime managment
  // syn_tensors are convinient to use with synapse API
  std::vector<synapse_helpers::tensor> tensor_helpers;
  std::vector<synTensor> syn_tensors;

  tensor_helpers.reserve(num_tensors);
  syn_tensors.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    tensor_helpers.push_back(
        habana_helpers::create_tensor(*tensors[i], names[i], persistents[i]));
    syn_tensors.push_back(tensor_helpers[i].get());
  }

  return {std::move(tensor_helpers), std::move(syn_tensors)};
}

std::vector<std::string> habana_helpers::names(const std::vector<synapse_helpers::tensor>& vec){
  std::vector<std::string> names;
  names.reserve(vec.size());

  std::transform(
      vec.begin(),
      vec.end(),
      std::back_inserter(names),
      [](auto& x) { return x.tensor_name_; });

  return names;
}
