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

#include "generated/lazy/frexp.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
template <>
LazyFrexp<std::tuple<at::Tensor, at::Tensor>>::LazyFrexp(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyFrexp<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto t = get_inputs().at(0).toTensor();
  c10::IntArrayRef out_shape = t.sizes();
  at::Tensor mantissa = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  at::Tensor exponent = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Int),
      t.suggest_memory_format(),
      false);
  return {mantissa, exponent};
}

} // namespace habana
