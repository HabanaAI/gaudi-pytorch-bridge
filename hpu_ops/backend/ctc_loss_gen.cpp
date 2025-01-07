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

#include "generated/backend/_ctc_loss.h"

namespace habana {

OutputMetaDataVector CtcLossMeta(const at::Stack& stack) {
  auto log_probs = stack_tensor(stack, 0);
  auto log_probs_sizes = log_probs.sizes(); // (T, N, C) or (T, C)
  auto targets_sizes = stack_tensor(stack, 1).sizes(); // (N, S)

  int64_t input_sequence_length = log_probs_sizes.at(0);
  int64_t batch_size = log_probs_sizes.at(1);
  int64_t max_target_length =
      targets_sizes.size() > 1 ? targets_sizes.at(1) : targets_sizes.at(0);

  OutputMetaData meta_n;
  meta_n.dtype = log_probs.scalar_type();
  meta_n.shape = std::vector<int64_t>{batch_size}; // N

  // Support for one output loss with reduction input for ctc_loss.Tensor
  if (stack.size() > 6) {
    auto reduction = stack.at(5).toInt();
    if (reduction != 0)
      meta_n.shape = std::vector<int64_t>{1};

    return {meta_n};
  } else {
    OutputMetaData meta_tns;
    meta_tns.dtype = log_probs.scalar_type();
    meta_tns.shape = std::vector<int64_t>{
        input_sequence_length,
        batch_size,
        2 * max_target_length + 1}; // (T, N, 2*S+1)

    return {meta_n, meta_tns};
  }
}

SharedMetaDataVector CtcLossSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& logProbs = stack_tensor(stack, 0);
  const auto& targets = stack_tensor(stack, 1);
  const auto dtype = logProbs.scalar_type();

  SharedMetaData ctcLossSharedMeta{"ctc_loss_fwd"};
  ctcLossSharedMeta.inputs_data = {
      {logProbs.dim(), dtype}, {targets.dim(), targets.scalar_type()}};
  if (stack.at(2).isTensor()) {
    const auto& inputLengths = stack_tensor(stack, 2);
    const auto& targetsLengths = stack_tensor(stack, 3);
    ctcLossSharedMeta.inputs_data.emplace_back(
        inputLengths.dim(), inputLengths.scalar_type());
    ctcLossSharedMeta.inputs_data.emplace_back(
        targetsLengths.dim(), targetsLengths.scalar_type());
  } else {
    ctcLossSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
    ctcLossSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
  }

  ctcLossSharedMeta.outputs_data.emplace_back(1, c10::ScalarType::Float);
  if (stack.size() < 7)
    ctcLossSharedMeta.outputs_data.emplace_back(3, dtype);

  return {ctcLossSharedMeta};
}

void CtcLoss::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto log_probs = stack_tensor(stack, 0);
  auto targets = stack_tensor(stack, 1);
  auto blank_index = stack.at(4).toInt();

  bool zero_infinity{};
  LossMode_t reduction_mode{LossMode_t::LOSS_REDUCTION_MODE_NONE};

  // Support for one output loss with reduction input for ctc_loss.Tensor
  bool oneOutputOnly = stack.size() > 6;

  if (oneOutputOnly) {
    zero_infinity = stack.at(6).toBool();

    auto reduction = stack.at(5).toInt();
    if (reduction == 1)
      reduction_mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
    else if (reduction == 2)
      reduction_mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
  } else {
    zero_infinity = stack.at(5).toBool();
  }

  ns_CTCLoss::Params params;
  params.blankIndex = blank_index;
  params.reductionMode = reduction_mode;
  params.zeroInfinity = zero_infinity;

  update_guid_dtype(guid_, log_probs.scalar_type());

  OutputMetaDataVector output_shapes = CtcLossMeta(stack);

  std::vector<synTensor> inputs{syn_in(0), syn_in(1)};
  std::vector<synapse_helpers::tensor> inputs_tensor;

  if (stack[2].isTensor()) {
    inputs.push_back(syn_in(2));
    inputs.push_back(syn_in(3));
  } else if (!isOutputInfMode()) {
    auto input_lengths = stack.at(2).toIntList().vec();
    auto target_lengths = stack.at(3).toIntList().vec();

    inputs_tensor.push_back(AllocateConstantSynapseTensor<int64_t>(
        graph,
        p_context_->device_id_,
        input_lengths,
        at::OptionalIntArrayRef{}));
    inputs.push_back(inputs_tensor.back().get());
    inputs_tensor.push_back(AllocateConstantSynapseTensor<int64_t>(
        graph,
        p_context_->device_id_,
        target_lengths,
        at::OptionalIntArrayRef{}));
    inputs.push_back(inputs_tensor.back().get());
  }

  if (oneOutputOnly) {
    auto op = OpBackend::BuildNode(
        this,
        graph,
        {guid_,
         inputs,
         {{output_shapes[0].shape, c10::ScalarType::Float, 0}},
         &params,
         sizeof(params)});

    syn_out(0) = std::move(op[0]); // loss
  } else {
    auto op = OpBackend::BuildNode(
        this,
        graph,
        {guid_,
         inputs,
         {{output_shapes[0].shape, c10::ScalarType::Float, 0},
          {output_shapes[1].shape, targets.scalar_type(), 1}},
         &params,
         sizeof(params)});

    syn_out(0) = std::move(op[0]); // loss
    syn_out(1) = std::move(op[1]); // alpha
  }
}

} // namespace habana
