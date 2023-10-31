/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/binary_cross_entropy.h"
#include "generated/backend/binary_cross_entropy_backward.h"
#include "generated/backend/binary_cross_entropy_with_logits.h"
#include "hpu_ops/op_backend.h"

constexpr int64_t index_of_fwd_mode = 3;
constexpr int64_t index_of_fwd_self = 0;
constexpr int64_t index_of_fwd_reduction = 4;
constexpr int64_t index_of_bwd_self = 1;
namespace habana {

OutputMetaDataVector BinaryCrossEntropyFwdMetaData(const at::Stack& stack) {
  auto self = stack.at(index_of_fwd_self).toTensor();
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  auto reduction = stack.at(index_of_fwd_mode).toInt();
  if (reduction == at::Reduction::Reduction::None)
    meta.shape = self.sizes().vec();
  else
    meta.shape = {};
  return {meta};
}

OutputMetaDataVector BinaryCrossEntropyLogitsFwdMetaData(
    const at::Stack& stack) {
  auto self = stack.at(index_of_fwd_self).toTensor();
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  auto reduction = stack.at(index_of_fwd_reduction).toInt();
  if (reduction == at::Reduction::Reduction::None)
    meta.shape = self.sizes().vec();
  else
    meta.shape = {};
  return {meta};
}

OutputMetaDataVector BinaryCrossEntropyBwdMetaData(const at::Stack& stack) {
  auto self = stack.at(index_of_bwd_self).toTensor();
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return {meta};
}

static std::shared_ptr<void> BceParams(
    const at::Stack& stack,
    size_t& size,
    const bool is_weights_used,
    const int reduction_index,
    const bool is_binary_cross_entropy_without_sigmoid,
    const PosWeightMode_t pos_mode) {
  PARAMS_STUB(ns_BinaryCrossEntropy::ParamsOptionalPosWeight);
  auto mode = stack.at(reduction_index).toInt();
  params->isWeightsUsed = is_weights_used;
  params->binaryCrossEntropyWithoutSigmoid =
      is_binary_cross_entropy_without_sigmoid;
  params->posMode = pos_mode;
  switch (mode) {
    case at::Reduction::Reduction::None:
      params->mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_NO_REDUCTION;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_SUM;
      break;
    default:
      TORCH_CHECK(
          false, "Unsupported reduction mode in Binarycrossentropy: ", mode);
  }
  return params;
}

// Forward variant
void BinaryCrossEntropyFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = BinaryCrossEntropyFwdMetaData(stack)[0].shape;
  const bool is_weights_used = !stack.at(2).isNone();
  const int reduction_index = 3;
  const bool is_binary_cross_entropy_without_sigmoid = true;
  const PosWeightMode_t pos_mode = PosWeightMode_t::POS_WEIGHT_DISABLE;
  size_t size = 0;
  auto params = BceParams(
      stack,
      size,
      is_weights_used,
      reduction_index,
      is_binary_cross_entropy_without_sigmoid,
      pos_mode);

  std::vector<synTensor> input{syn_in(0), syn_in(1)};
  std::vector<synapse_helpers::tensor> weight;
  if (is_weights_used) {
    std::vector<int64_t> target_shape = stack.at(1).toTensor().sizes().vec();
    auto broadcast_weight =
        BroadcastHelper(graph, syn_in(2), target_shape, ScalarType());
    weight.emplace_back(std::move(broadcast_weight));
    input.emplace_back(weight[0].get());
  }

  auto bce_logits_fwd = BuildOp(
      graph,
      get_guid_with_precision("binary_cross_entropy_fwd", ScalarType()),
      std::move(input),
      {{output_shape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(bce_logits_fwd[0]);
}

// Forward variant
void BinaryCrossEntropyWithLogitsFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = BinaryCrossEntropyLogitsFwdMetaData(stack)[0].shape;
  const bool is_weights_used = !stack.at(2).isNone();
  const bool is_pos_weights_used = !stack.at(3).isNone();
  const int reduction_index = 4;
  const bool is_binary_cross_entropy_without_sigmoid = false;
  const PosWeightMode_t pos_mode = is_pos_weights_used
      ? PosWeightMode_t::POS_WEIGHT_ENABLE
      : PosWeightMode_t::POS_WEIGHT_DISABLE;
  size_t size = 0;
  auto params = BceParams(
      stack,
      size,
      is_weights_used,
      reduction_index,
      is_binary_cross_entropy_without_sigmoid,
      pos_mode);

  std::vector<synTensor> input{syn_in(0), syn_in(1)};
  if (is_pos_weights_used) {
    if (is_weights_used)
      input.emplace_back(syn_in(3));
    else
      input.emplace_back(syn_in(2));
  }
  std::vector<synapse_helpers::tensor> weight;
  if (is_weights_used) {
    std::vector<int64_t> target_shape = stack.at(1).toTensor().sizes().vec();
    auto broadcast_weight =
        BroadcastHelper(graph, syn_in(2), target_shape, ScalarType());
    weight.emplace_back(std::move(broadcast_weight));
    input.emplace_back(weight[0].get());
  }

  auto bce_logits_fwd = BuildOp(
      graph,
      get_guid_with_precision("binary_cross_entropy_fwd", ScalarType()),
      std::move(input),
      {{output_shape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(bce_logits_fwd[0]);
}

// Backward variant
void BinaryCrossEntropyBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr int64_t index_of_grad = 0;
  constexpr int64_t index_of_self = 1;
  constexpr int64_t index_of_target = 2;

  auto bce_output_shape = BinaryCrossEntropyBwdMetaData(stack)[0].shape;
  const bool is_weights_used = !stack.at(3).isNone();
  const int reduction_index = 4;
  const bool is_binary_cross_entropy_without_sigmoid = true;
  const PosWeightMode_t pos_mode = PosWeightMode_t::POS_WEIGHT_DISABLE;
  size_t size = 0;
  auto params = BceParams(
      stack,
      size,
      is_weights_used,
      reduction_index,
      is_binary_cross_entropy_without_sigmoid,
      pos_mode);

  auto neg_grad = BuildOp(
      graph,
      get_guid_with_precision("neg_fwd", ScalarType()),
      {syn_in(index_of_grad)},
      {{stack.at(index_of_grad).toTensor().sizes().vec(), ScalarType()}});

  std::vector<synTensor> bce_bwd_inputs = {
      syn_in(index_of_self), syn_in(index_of_target)};

  if (is_weights_used) {
    bce_bwd_inputs.push_back(syn_in(3));
  }
  bce_bwd_inputs.push_back(neg_grad[0].get());

  auto bce_bwd = BuildOp(
      graph,
      get_guid_with_precision("binary_cross_entropy_bwd", ScalarType()),
      std::move(bce_bwd_inputs),
      {{bce_output_shape, ScalarType(), 0}},
      params.get(),
      size);

  // output of bce_bwd is the output of this op
  syn_out(0) = std::move(bce_bwd[0]);
}
} // namespace habana
