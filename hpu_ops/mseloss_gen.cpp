/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

namespace habana {
sizes_vec HabanaOperatorHelper::MseLossOutputShape(
    const torch::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return {self.sizes().vec()};
  }
  return {{}};
}

std::shared_ptr<void> HabanaOperatorHelper::FillMseLossParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MSELossKernel::Params);

  auto mode = stack.at(stack.at(2).isInt() ? 2 : 3).toInt();
  switch (mode) {
    case at::Reduction::Reduction::None:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_NONE;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_SUM;
      break;
    default:
      TORCH_CHECK(false, "Unsupported reduction mode in mseloss: ", mode);
  }
  return params;
}

} // namespace habana
