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
  TORCH_CHECK(
      result != map.end(), "Unsupported pytorch type ", pt_type);

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
