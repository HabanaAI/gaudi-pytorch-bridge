/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/eager/max_pool3d_with_indices.h"
#include "generated/lazy/max_pool2d_with_indices.h"
#include "generated/lazy/max_pool2d_with_indices_backward.h"

namespace habana {

template <>
LazyMaxPool<std::tuple<at::Tensor, at::Tensor>>::LazyMaxPool(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<std::tuple<at::Tensor, at::Tensor>>(
          qualstring,
          inputs,
          out_shapes_fn) {}

template <>
std::tuple<at::Tensor, at::Tensor> LazyMaxPool<
    std::tuple<at::Tensor, at::Tensor>>::get_result_overrideable() {
  auto inputs = get_inputs();
  auto t = inputs.at(0).toTensor();
  auto out_shape = get_out_shapes()[0];
  at::Tensor maxpool = habana_lazy::empty_hpu_lazy(
      out_shape, t.options(), t.suggest_memory_format(), false);
  // TODO Analyse memory performance for the dtype change
  // (https://jira.habana-labs.com/browse/SW-108396),
  at::Tensor indices = habana_lazy::empty_hpu_lazy(
      out_shape,
      t.options().dtype(c10::ScalarType::Long),
      t.suggest_memory_format(),
      false);
  return {maxpool, indices};
}

} // namespace habana
