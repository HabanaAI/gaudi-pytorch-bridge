/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/common/add_composite_gen.h"
#include "generated/lazy/addcdiv.h"
#include "generated/lazy/addcmul.h"

namespace habana {

static void convert_scalar_val_to_tensor(std::vector<at::IValue>& inputs) {
  auto self = inputs[inp_idx].toTensor();
  auto s = inputs[val_idx].toScalar();
  double val = s.to<double>();
  at::Tensor val_t;
  if (val != 1.0)
    val_t = habana_lazy::get_tensor_for_scalar(val, self.options());
  c10::optional<at::Tensor> val_t_opt = c10::make_optional(val_t);
  inputs[val_idx] = val_t_opt;
}

template <>
AddCOpFE<at::Tensor&>::AddCOpFE(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  // convert "value" scalar to tensor to avoid cache misses
  convert_scalar_val_to_tensor(get_inputs());
}

template <>
AddCOpFE<at::Tensor>::AddCOpFE(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  // convert "value" scalar to tensor to avoid cache misses
  convert_scalar_val_to_tensor(get_inputs());
}
template <>
at::Tensor& AddCOpFE<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}
template <>
at::Tensor AddCOpFE<at::Tensor>::get_result_overrideable() {
  return LazyOp<at::Tensor>::get_result_overrideable();
}
} // namespace habana
