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
#include "hpu_ops/common/reduction_template.h"

namespace habana {

template <>
at::Tensor ReductionFrontendTemplate<at::Tensor>::get_result_overrideable() {
  const auto& stack = LazyOp<at::Tensor>::get_inputs();
  const torch::Tensor& self = stack_tensor(stack, 0);

  return at::native::create_reduction_result(
      self,
      get_dims(stack, m_dim_index),
      get_keepdim(stack, m_keepdim_index),
      get_scalar_type());
}

template <>
at::Tensor& ReductionFrontendTemplate<at::Tensor&>::get_result_overrideable() {
  throw std::invalid_argument("Tensor ref should not be created.");
}

} // namespace habana
