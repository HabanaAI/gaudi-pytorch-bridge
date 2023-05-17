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
#include "hpu_ops/lazy/reduction_template.h"

namespace habana {

template <>
at::Tensor ReductionFrontendTemplate<at::Tensor>::get_result_overrideable() {
  auto inputs = get_inputs();
  auto self = inputs[0].toTensor();
  auto dim = get_dims(inputs, m_dim_index);
  auto keepdim = get_keepdim(inputs, m_keepdim_index);
  at::native::DimMask mask = at::native::make_dim_mask(dim, self.dim());
  auto shape = at::native::shape_from_dim_mask(self, mask, keepdim);
  return habana_lazy::empty_hpu_lazy(
      shape,
      self.options().dtype(get_scalar_types()[0]),
      self.suggest_memory_format(),
      false);
}

template <>
at::Tensor& ReductionFrontendTemplate<at::Tensor&>::get_result_overrideable() {
  throw std::invalid_argument("Tensor ref should not be created.");
}

} // namespace habana
