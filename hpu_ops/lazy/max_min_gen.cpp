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

#include "generated/lazy/max.h"
#include "generated/lazy/min.h"
#include "hpu_ops/lazy/reduction_template.h"

namespace habana {

template <>
LazyMinMax<std::tuple<at::Tensor, at::Tensor>>::LazyMinMax(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn,
          -1) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyMinMax<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto inputs = get_inputs();
  auto t = inputs.at(0).toTensor();
  auto out_shape = MinMaxOutputShape(inputs)[0];
  at::Tensor min = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  at::Tensor min_indices = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);
  return {min, min_indices};
}
} // namespace habana
