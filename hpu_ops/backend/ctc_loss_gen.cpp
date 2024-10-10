/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
