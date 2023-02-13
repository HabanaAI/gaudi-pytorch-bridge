/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <torch/script.h>

#include "backend/helpers/cast_sequence.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/frontend_utils.h"
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
  std::vector<at::Tensor> pt_inputs{self};
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    std::vector<at::Tensor> v{self};
    Op.Execute(key, pt_inputs, v);
  } else {
    // Ideally _out version of operator does not need inputs
    // but in this case we are giving an input to align with
    // graph mode behavior. Internally within ConstantOut
    // implementation we will move input tensors to output tensors
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
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
