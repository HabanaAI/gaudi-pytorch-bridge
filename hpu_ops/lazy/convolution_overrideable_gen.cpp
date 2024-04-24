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
#include "generated/lazy/convolution_overrideable.h"
#include "habana_lazy/permute_tensors.h"
#include "habana_lazy/view_utils.h"

namespace habana {

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(
    habana_lazy::LazyOp,
    ConvolutionOverrideableFE,
    at::Tensor) {
  const auto weight = inputs[1].toTensor();

  const auto hl_t = habana_lazy::GetHbLazyTensor(weight);
  const bool is_view_tensor = hl_t.getDataPtr()->stride_params.has_value();

  if (!is_view_tensor) {
    at::Tensor weight_hpu =
        habana_lazy::HbLazyTensorViews::HandleViewsD2H(weight);

    habana_lazy::PermuteTensors::permuteWeight(weight_hpu);
  }
}

} // namespace habana
