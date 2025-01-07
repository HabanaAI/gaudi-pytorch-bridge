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

#include "generated/backend/_ctc_loss_backward.h"

namespace habana {

OutputMetaDataVector CtcLossBackwardMeta(const at::Stack& stack) {
  const auto log_probs = stack_tensor(stack, 1);
  std::vector<int64_t> log_probs_shape = log_probs.sizes().vec();

  OutputMetaData meta;
  meta.dtype = log_probs.scalar_type();
  meta.shape = log_probs_shape;

  return {meta};
}

SharedMetaDataVector CtcLossBackwardSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& logProbs = stack_tensor(stack, 1);
  const auto& targets = stack_tensor(stack, 2);
  const auto& negLogLikelihood = stack_tensor(stack, 5);
  const auto& logAlpha = stack_tensor(stack, 6);

  SharedMetaData ctcLossBackwardSharedMeta{"ctc_loss_bwd"};
  ctcLossBackwardSharedMeta.inputs_data = {
      {grad.dim(), grad.scalar_type()},
      {logProbs.dim(), logProbs.scalar_type()},
      {targets.dim(), targets.scalar_type()}};
  if (stack.at(3).isTensor()) {
    const auto& inputLengths = stack_tensor(stack, 3);
    const auto& targetsLengths = stack_tensor(stack, 4);
    ctcLossBackwardSharedMeta.inputs_data.emplace_back(
        inputLengths.dim(), inputLengths.scalar_type());
    ctcLossBackwardSharedMeta.inputs_data.emplace_back(
        targetsLengths.dim(), targetsLengths.scalar_type());
  } else {
    ctcLossBackwardSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
    ctcLossBackwardSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
  }
  ctcLossBackwardSharedMeta.inputs_data.emplace_back(
      negLogLikelihood.dim(), negLogLikelihood.scalar_type());
  ctcLossBackwardSharedMeta.inputs_data.emplace_back(
      logAlpha.dim(), logAlpha.scalar_type());
  ctcLossBackwardSharedMeta.outputs_data.emplace_back(
      logProbs.dim(), logProbs.scalar_type());

  return {ctcLossBackwardSharedMeta};
}

void CtcLossBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto grad = stack_tensor(stack, 0);
  auto log_probs = stack_tensor(stack, 1);
  auto targets = stack_tensor(stack, 2);
  auto neg_log_likelihood = stack_tensor(stack, 5); // loss from fwd
  auto log_alpha = stack_tensor(stack, 6); // alpha from fwd
  auto blank_index = stack.at(7).toInt();
  bool zero_infinity = stack.at(8).toBool();

  ns_CTCLoss::Params params;
  params.blankIndex = blank_index;
  params.reductionMode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
  params.zeroInfinity = zero_infinity;

  update_guid_dtype(guid_, log_probs.scalar_type());

  std::vector<synTensor> inputs{syn_in(0), syn_in(1), syn_in(2)};
  std::vector<synapse_helpers::tensor> inputs_tensor;

  if (stack[3].isTensor()) {
    for (size_t i = 3; i <= 6; ++i)
      inputs.push_back(syn_in(i));
  } else if (!isOutputInfMode()) {
    auto input_lengths = stack.at(3).toIntList().vec();
    auto target_lengths = stack.at(4).toIntList().vec();

    inputs_tensor.push_back(AllocateConstantSynapseTensor(
        graph,
        p_context_->device_id_,
        input_lengths,
        at::OptionalIntArrayRef{}));
    inputs.push_back(inputs_tensor.back().get());
    inputs_tensor.push_back(AllocateConstantSynapseTensor(
        graph,
        p_context_->device_id_,
        target_lengths,
        at::OptionalIntArrayRef{}));
    inputs.push_back(inputs_tensor.back().get());

    inputs.push_back(syn_in(3));
    inputs.push_back(syn_in(4));
  }

  auto op = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       inputs,
       {{log_probs.sizes(), log_probs.scalar_type(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(op[0]); // gradOut
}

} // namespace habana
