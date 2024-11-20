
/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

SharedMetaDataVector SoftmaxBackwardSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& gradOutput = stack_tensor(stack, 0);
  const auto& output = stack_tensor(stack, 1);
  const auto dtype = output.scalar_type();
  const auto rank = gradOutput.dim();

  SharedMetaData softmaxBwdSharedMeta{"softmax_bwd"};
  softmaxBwdSharedMeta.inputs_data = {
      {output.dim(), dtype}, {rank, gradOutput.scalar_type()}};
  softmaxBwdSharedMeta.outputs_data.emplace_back(rank, dtype);

  return {softmaxBwdSharedMeta};
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
