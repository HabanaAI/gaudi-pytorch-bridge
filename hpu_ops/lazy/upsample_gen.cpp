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

#include "generated/lazy/upsample_linear1d.h"
#include "generated/lazy/upsample_linear1d_backward.h"
#include "generated/lazy/upsample_nearest1d.h"
#include "generated/lazy/upsample_nearest1d_backward.h"
#include "generated/lazy/upsample_nearest2d.h"
#include "generated/lazy/upsample_nearest2d_backward.h"
#include "generated/lazy/upsample_nearest3d.h"
#include "generated/lazy/upsample_nearest3d_backward.h"

namespace habana {
// FrontEnd
template <>
LazyUpsample<at::Tensor>::LazyUpsample(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  auto x = get_inputs();
  auto self = x[0].toTensor();
  auto meta = UpsampleNearest2DBwdMeta(x)[0];
  auto outshape_st = habana_lazy::empty_hpu_lazy(
      meta.shape,
      self.options(),
      self.suggest_memory_format(),
      false,
      SHAPE_TENSOR);

  x[2] = c10::IValue(outshape_st);
  set_inputs(x);
}

template <>
at::Tensor LazyUpsample<at::Tensor>::get_result_overrideable() {
  return LazyOp<at::Tensor>::get_result_overrideable();
}

template struct LazyUpsample<at::Tensor>;
} // namespace habana
