#include "kernel_utils.h"

std::string habana_helpers::unique_recipe_name_generator(
    std::string recipe_name) {
  static std::unordered_map<std::string, unsigned> map;
  return recipe_name + std::to_string(map[recipe_name]++);
}

synTensorDescriptorTr habana_helpers::synapse_tensor_descriptor_builder(
    const c10::IntArrayRef& shape,
    const synDataType dtype,
    const std::string& name,
    const bool persistent) {
  synTensorDescriptorTr descriptor;
  descriptor.m_dataType = dtype;
  descriptor.m_dims = shape.size();

  TORCH_CHECK(
      shape.size() <= SYN_MAX_TENSOR_DIM,
      name,
      " tensor has more than ",
      SYN_MAX_TENSOR_DIM,
      " dimensions");
  // write NHWC as CWHN and write 0 at the end
  for (int i = 0; i < shape.size(); ++i)
    descriptor.m_sizes[i] = shape[shape.size() - i - 1];
  for (int i = shape.size(); i < SYN_MAX_TENSOR_DIM; ++i)
    descriptor.m_sizes[i] = 0;

  //   descriptor.m_strides[SYN_MAX_TENSOR_DIM]; // TODO: not needed?
  descriptor.m_name = name.c_str(); // TODO: we take only pointer, so make sure
                                    // name object will be alive
  descriptor.m_deviceMemAddress = 0; // It will be patched during runtime
  descriptor.m_isOutput = persistent;
  descriptor.m_isPersistent = persistent;

  return descriptor;
}