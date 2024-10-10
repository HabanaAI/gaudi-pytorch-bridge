/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/eager/reduction_template.h"
#include "hpu_ops/common/reduction_template.h"

namespace habana {

template <>
at::Tensor ReductionFrontendTemplate<at::Tensor>::get_result_overrideable() {
  auto inputs = get_inputs();
  auto self = inputs[0].toTensor();
  auto dim = get_dims(inputs, m_dim_index);
  auto keepdim = get_keepdim(inputs, m_keepdim_index);
  return at::native::create_reduction_result(
      self, dim, keepdim, get_scalar_types()[0]);
}

template <>
at::Tensor& ReductionFrontendTemplate<at::Tensor&>::get_result_overrideable() {
  throw std::invalid_argument("Tensor ref should not be created.");
}

} // namespace habana
