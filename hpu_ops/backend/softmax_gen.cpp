
/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/_softmax_backward_data.h"

namespace habana {
std::shared_ptr<void> FillSoftmaxForwardParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_Softmax::Params);

  // index positions for input args
  constexpr size_t selfPositionInArgList = 0;
  constexpr size_t dimPositionInArgList = 1;
  constexpr size_t halfToFloatPositionInArgList = 2;

  auto self = stack.at(selfPositionInArgList).toTensor();
  int dim = stack.at(dimPositionInArgList).toInt();
  bool half_to_float = stack.at(halfToFloatPositionInArgList).toBool();
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");
  params->dim = get_dim_in_tpc_order(dim, self.dim());
  return params;
}

std::shared_ptr<void> FillSoftmaxBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_Softmax::Params);
  // index positions for input args
  constexpr size_t selfPositionInArgList = 0;
  constexpr size_t dimPositionInArgList = 2;

  auto self = stack.at(selfPositionInArgList).toTensor();
  int dim = stack.at(dimPositionInArgList).toInt();
  params->dim = get_dim_in_tpc_order(dim, self.dim());
  return params;
}

void SoftmaxBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr size_t selfPositionInArgList = 0;
  const auto& outshape = stack_tensor(stack, selfPositionInArgList).sizes();

  size_t size = 0;
  const auto& params = FillSoftmaxBackwardParams(stack, size);

  auto softmax_bwd = BuildOp(
      graph,
      get_guid_with_precision("softmax_bwd", ScalarType()),
      {syn_in(1), syn_in(0)},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);

  // output of softmax_bwd is the output of this op
  syn_out(0) = std::move(softmax_bwd[0]);
}
} // namespace habana
