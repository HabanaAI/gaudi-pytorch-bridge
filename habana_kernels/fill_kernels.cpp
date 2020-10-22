/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"

using namespace torch;

template <typename T>
void synapse_fill(const Tensor& output, const T val) {
  // Using below approach of filling a buffer on HOST and then copying
  // to Device memory instead of doing a synMemSetD[]Async due to SW-11757
  // TODO revert to synMemSet once SW-11757 is resolved
  auto size = output.numel() * output.element_size();
  std::vector<T> buffer(size, val);

  habana_helpers::copy_scalar_to_device(buffer.data(), output, size);
}

/**
 * @brief This function uses "constant" TPC kernel to fill input
 * tensor with "value" provided. fp32, bf16 & i32 are the only
 * support dtypes. Tensor shall be filled with values based on
 * tensor's scalar type (value shall be casted to scalar type of
 * tensor if it happens to be different)
 */
void fill_constant_hpu(Tensor& self, Scalar value) {
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "constant_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Note that we fill the tensor based on its own scalar_type
  // and not based on dtype of value
  ConstantOutOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {IValue(self), IValue(value)};
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTOutputs({self});
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
}

Tensor& fill_hpu_(Tensor& self, Scalar value) {
  PT_KERNEL_BEGIN;
  auto dtype = habana_helpers::scalar_type(value);

  TORCH_CHECK(dtype != c10::ScalarType::Bool);

  switch (self.element_size()) {
    case 1: {
      TORCH_CHECK(value.isIntegral(false));
      auto memset_val = value.to<unsigned char>();
      synapse_fill(self, memset_val);
    } break;
    case 2: {
      if (self.scalar_type() == c10::ScalarType::BFloat16) {
        fill_constant_hpu(self, value);
      } else {
        auto memset_val = value.to<int16_t>();
        synapse_fill(self, memset_val);
      }
    } break;
    case 4:
      fill_constant_hpu(self, value);
      break;
    case 8: {
      // Even though HPU doesnt support long/double. Intermediate tensors in
      // embedding_bag used by PyT needs this fill functionality
      if (value.isIntegral(true)) {
        uint64_t memset_val = value.to<long>();
        synapse_fill(self, memset_val);
      } else {
        // double
        double memset_val = value.to<double>();
        synapse_fill(self, memset_val);
      }
    } break;
    default:
      PT_KERNEL_WARN("Unsupported data type used in fill");
  }
  PT_KERNEL_END;
  return self;
}

/** @brief Function implementing torch.Tensor.masked_fill_(mask, value)
 * @param self: (fp32/bf16, 1-4D) Input tensor
 * @param mask: (BoolTensor) the boolean mask
 * @param value: (floatTensor, 0D) the value to fill with
 */
Tensor& masked_fill_hpu_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(
      value.dim() == 0, "value supports only 0D tensor to match CPU behavior");

  auto mask_expand = mask;
  if (self.sizes() != mask.sizes()) {
    // this explicit broadcast can be removed when
    // binary kernels start supporting broadcase
    mask_expand = mask.expand(self.sizes());
  }

  TORCH_CHECK(
      self.sizes() == mask_expand.sizes(),
      "input & mask tensor shapes not matching");

  // mask (datatype "bool") needs to be casted because TPC kernels support
  // fp32/bf16 only
  auto new_mask = habana_helpers::hpu_cast_tensor(mask_expand, self.dtype());
  // create a inverted mask
  auto zero_tensor = at::zeros_like(
      new_mask, new_mask.options(), new_mask.suggest_memory_format());
  auto inv_mask = habana_helpers::hpu_cast_tensor(
      at::eq(new_mask, zero_tensor), self.dtype());

  // broadcast value to same shape as input tensor
  // this explicit broadcast can be removed when
  // binary kernels start supporting broadcase
  auto value_expand = value.expand(self.sizes());

  // mask_fill computation
  self.mul_(inv_mask);
  self.add_(new_mask * value_expand);

  PT_KERNEL_END;
  return self;
}

/** @brief Function implementing torch.Tensor.masked_fill_(mask, value)
 * @param self: (fp32/bf16, 1-4D) Input tensor
 * @param mask: (BoolTensor) the boolean mask
 * @param value: (float) the value to fill with
 */
Tensor& masked_fill_scalar_hpu_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  // convert scalar fill value to device tensor
  auto value_tensor = habana_helpers::scalar_to_device_tensor(value, self, 0);

  return masked_fill_hpu_(self, mask, value_tensor);
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::fill_",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<ConstantOutOperator>(device_id, node_type);
    });