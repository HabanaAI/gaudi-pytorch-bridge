/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/_log_softmax_backward_data.h"

namespace habana {
std::shared_ptr<void> FillLogSoftmaxParams(
    const at::Stack& stack,
    size_t& size) {
  bool half_to_float = stack.at(2).toBool();
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");
  auto self = stack.at(0).toTensor();
  PARAMS_STUB(ns_Softmax::Params);
  params->dim = get_dim_in_tpc_order(
      /*dim*/ stack.at(1).toInt(),
      /*max dims*/ self.dim());
  return params;
}

std::shared_ptr<void> FillLogSoftmaxBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  PARAMS_STUB(ns_Softmax::Params);
  params->dim = get_dim_in_tpc_order(stack.at(2).toInt(), self.dim());
  return params;
}

} // namespace habana
