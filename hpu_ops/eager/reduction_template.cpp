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
  return CommonReductionFrontendTemplate::CreateResult(
      EagerOp<at::Tensor>::get_inputs(), get_scalar_types()[0]);
}

template <>
at::Tensor& ReductionFrontendTemplate<at::Tensor&>::get_result_overrideable() {
  throw std::invalid_argument("Tensor ref should not be created.");
}

} // namespace habana
