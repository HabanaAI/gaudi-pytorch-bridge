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
#include "hpu_op_helper.h"

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

void LogSoftmaxBackward::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  constexpr int inputTensorPos = 3;
  const auto& outshape = stack_tensor(stack, inputTensorPos).sizes();

  size_t size = 0;
  const auto& params = FillLogSoftmaxBackwardParams(stack, size);
  auto log_softmax_bwd = BuildOp(
      graph,
      guid_,
      {syn_in(1), syn_in(0)},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}},
      params.get(),
      size);

  syn_out(0) = std::move(log_softmax_bwd[0]);
}
} // namespace habana
