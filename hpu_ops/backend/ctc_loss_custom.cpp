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

#include "hpu_ops/ctc_loss_custom.h"

namespace habana {

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
calculate_output_shapes_for_ctc_loss_custom_fwd(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const int64_t reduction) {
  auto log_probs_sizes = log_probs.sizes(); // (T, N, C) or (T, C)
  auto targets_sizes = targets.sizes(); // (N, S)

  int64_t input_sequence_length = log_probs_sizes.at(0);
  int64_t batch_size = log_probs_sizes.size() > 2 ? log_probs_sizes.at(1) : 1;
  int64_t max_target_length =
      targets_sizes.size() > 1 ? targets_sizes.at(1) : targets_sizes.at(0);

  auto loss_shape = std::vector<int64_t>{};
  if (reduction == 0) {
    loss_shape = std::vector<int64_t>{batch_size};
  }

  auto alpha_shape = std::vector<int64_t>{
      input_sequence_length,
      batch_size,
      2 * max_target_length + 1}; // (T, N, 2*S+1)

  return std::make_tuple(loss_shape, alpha_shape);
}

OutputMetaDataVector CTCLossCustomMeta(const at::Stack& stack) {
  auto log_probs = stack_tensor(stack, 0);
  auto targets = stack_tensor(stack, 1);
  auto shapes = calculate_output_shapes_for_ctc_loss_custom_fwd(
      log_probs, targets, stack.at(5).toInt());

  OutputMetaData meta_loss;
  meta_loss.dtype = log_probs.scalar_type();
  meta_loss.shape = std::get<0>(shapes);

  OutputMetaData meta_alpha;
  meta_alpha.dtype = log_probs.scalar_type();
  meta_alpha.shape = std::get<1>(shapes);

  return {meta_loss, meta_alpha};
}

CTCLossCustom::CTCLossCustom(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "ctc_loss_fwd", scalar_type, {0, 0}, {}, {}, false) {
  SetOutputMetaFn(CTCLossCustomMeta);
}

void CTCLossCustom::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto log_probs = stack_tensor(stack, 0);
  auto targets = stack_tensor(stack, 1);
  auto blank_index = stack.at(4).toInt();
  auto zero_infinity = stack.at(6).toBool();

  LossMode_t reduction_mode{LossMode_t::LOSS_REDUCTION_MODE_NONE};
  auto reduction = stack.at(5).toInt();
  if (reduction == 1)
    reduction_mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  else if (reduction == 2)
    reduction_mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;

  ns_CTCLoss::Params params;
  params.blankIndex = blank_index;
  params.reductionMode = reduction_mode;
  params.zeroInfinity = zero_infinity;

  update_guid_dtype(guid_, log_probs.scalar_type());

  auto meta = CTCLossCustomMeta(stack);

  std::vector<synTensor> inputs{};
  for (size_t i = 0; i < 4; ++i)
    inputs.push_back(syn_in(i));

  auto op = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{meta[0].shape, meta[0].dtype, 0}, {meta[1].shape, meta[1].dtype, 1}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(op[0]); // loss
  syn_out(1) = std::move(op[1]); // alpha
}

CTCLossCustomBackward::CTCLossCustomBackward(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(device_id, "ctc_loss_bwd", scalar_type, {1}, {}, {}, false) {}

void CTCLossCustomBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto grad = stack_tensor(stack, 0);
  auto log_probs = stack_tensor(stack, 1);
  auto targets = stack_tensor(stack, 2);
  auto neg_log_likelihood = stack_tensor(stack, 5); // loss from fwd
  auto log_alpha = stack_tensor(stack, 6); // alpha from fwd
  auto blank_index = stack.at(7).toInt();
  auto zero_infinity = stack.at(9).toBool();

  LossMode_t reduction_mode{LossMode_t::LOSS_REDUCTION_MODE_NONE};

  auto reduction = stack.at(8).toInt();
  if (reduction == 1)
    reduction_mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  else if (reduction == 2)
    reduction_mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;

  ns_CTCLoss::Params params;
  params.blankIndex = blank_index;
  params.reductionMode = reduction_mode;
  params.zeroInfinity = zero_infinity;

  update_guid_dtype(guid_, log_probs.scalar_type());

  std::vector<synTensor> inputs{};
  for (size_t i = 0; i < 7; ++i)
    inputs.push_back(syn_in(i));

  auto op = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{log_probs.sizes(), log_probs.scalar_type(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(op[0]); // gradOut
}

} // namespace habana

static const auto& CtcLossCustomKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::ctc_loss_custom", KERNEL_FN_GLOBAL(habana::CTCLossCustom))
        .add(
            "hpu::ctc_loss_custom_backward",
            KERNEL_FN_GLOBAL(habana::CTCLossCustomBackward));
