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
SharedMetaDataVector BinaryCrossEntropyFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack_tensor(stack, 0);
  auto target = stack_tensor(stack, 1);
  auto weights = stack.at(2).toOptional<at::Tensor>();
  auto reduction = stack.at(3).toInt();
  auto dtype = self.scalar_type();
  auto outputRank =
      reduction == at::Reduction::Reduction::None ? self.dim() : 1;

  SharedMetaData binaryCrossEntropyFwdSharedMeta{"binary_cross_entropy_fwd"};
  binaryCrossEntropyFwdSharedMeta.inputs_data = {
      {self.dim(), dtype}, {target.dim(), dtype}};
  if (weights.has_value())
    binaryCrossEntropyFwdSharedMeta.inputs_data.emplace_back(
        target.dim(), dtype);

  binaryCrossEntropyFwdSharedMeta.outputs_data.emplace_back(outputRank, dtype);
  return {binaryCrossEntropyFwdSharedMeta};
}

SharedMetaDataVector BinaryCrossEntropyWithLogitsFwdSharedMeta(
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto target = stack_tensor(stack, 1);
  auto weights = stack.at(2).toOptional<at::Tensor>();
  auto posWeights = stack.at(3).toOptional<at::Tensor>();

  auto reduction = stack.at(4).toInt();
  auto dtype = self.scalar_type();
  auto outputRank =
      reduction == at::Reduction::Reduction::None ? self.dim() : 1;

  SharedMetaData binaryCrossEntropyFwdSharedMeta{"binary_cross_entropy_fwd"};
  binaryCrossEntropyFwdSharedMeta.inputs_data = {
      {self.dim(), dtype}, {target.dim(), dtype}};
  if (posWeights.has_value())
    binaryCrossEntropyFwdSharedMeta.inputs_data.emplace_back(
        posWeights.value().dim(), dtype);

  if (weights.has_value())
    binaryCrossEntropyFwdSharedMeta.inputs_data.emplace_back(
        target.dim(), dtype);

  binaryCrossEntropyFwdSharedMeta.outputs_data.emplace_back(outputRank, dtype);
  return {binaryCrossEntropyFwdSharedMeta};
}

SharedMetaDataVector BinaryCrossEntropyBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto grad = stack_tensor(stack, 0);
  auto self = stack_tensor(stack, 1);
  auto target = stack_tensor(stack, 2);
  auto weights = stack.at(3).toOptional<at::Tensor>();
  auto rank = self.dim();
  auto dtype = self.scalar_type();

  SharedMetaData negGradSharedMeta{"neg_fwd"};
  negGradSharedMeta.inputs_data.emplace_back(grad.dim(), dtype);
  negGradSharedMeta.outputs_data = {negGradSharedMeta.inputs_data[0]};

  SharedMetaData binaryCrossEntropyBwdSharedMeta{"binary_cross_entropy_bwd"};
  binaryCrossEntropyBwdSharedMeta.inputs_data = {
      {rank, dtype}, {target.dim(), dtype}};
  if (weights.has_value())
    binaryCrossEntropyBwdSharedMeta.inputs_data.emplace_back(
        weights.value().dim(), dtype);

  binaryCrossEntropyBwdSharedMeta.inputs_data.push_back(
      negGradSharedMeta.outputs_data[0]);

  binaryCrossEntropyBwdSharedMeta.outputs_data.emplace_back(rank, dtype);

  return {negGradSharedMeta, binaryCrossEntropyBwdSharedMeta};
}

static std::shared_ptr<void> BceParams(
    const at::Stack& stack,
    size_t& size,
    const bool is_weights_used,
    const int reduction_index,
    const bool is_binary_cross_entropy_without_sigmoid,
    const PosWeightMode_t pos_mode) {
  PARAMS_STUB(ns_BinaryCrossEntropy::ParamsOptionalNormalize);
  params->isNormalizeWeights = 0;
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
  StackGetter stackGetter(this, stack, "BinaryCrossEntropyFwd::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto target = stackGetter.getNextInput<TensorsPair>();
  auto weights = stackGetter.getNextInput<c10::optional<TensorsPair>>();

  auto output_shape = BinaryCrossEntropyFwdMetaData(stack)[0].shape;
  const bool is_weights_used = weights.has_value();
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

  std::vector<synTensor> input{self.syn_t, target.syn_t};
  std::vector<synapse_helpers::tensor> weight;
  if (is_weights_used) {
    std::vector<int64_t> target_shape = target.pt_t.sizes().vec();
    auto broadcast_weight =
        BroadcastHelper(graph, weights->syn_t, target_shape, ScalarType());
    weight.emplace_back(std::move(broadcast_weight));
    input.emplace_back(weight[0].get());
  }

  auto bce_fwd = BuildOp(
      graph,
      get_guid_with_precision("binary_cross_entropy_fwd", ScalarType()),
      std::move(input),
      {{output_shape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(bce_fwd[0]);
}

// Forward variant
void BinaryCrossEntropyWithLogitsFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(
      this, stack, "BinaryCrossEntropyWithLogitsFwd::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto target = stackGetter.getNextInput<TensorsPair>();
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

  std::vector<synTensor> input{self.syn_t, target.syn_t};
  if (is_pos_weights_used) {
    if (is_weights_used)
      input.emplace_back(syn_in(3));
    else
      input.emplace_back(syn_in(2));
  }
  std::vector<synapse_helpers::tensor> weight;
  if (is_weights_used) {
    std::vector<int64_t> target_shape = target.pt_t.sizes().vec();
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
  StackGetter stackGetter(this, stack, "BinaryCrossEntropyBwd::AddNode");
  auto grad = stackGetter.getNextInput<TensorsPair>();
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto target = stackGetter.getNextInput<TensorsPair>();
  auto weights = stackGetter.getNextInput<c10::optional<TensorsPair>>();

  auto bce_meta = BinaryCrossEntropyBwdMetaData(stack)[0];
  const bool is_weights_used = weights.has_value();
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
      get_guid_with_precision("neg_fwd", bce_meta.dtype),
      {grad.syn_t},
      {{grad.pt_t.sizes().vec(), bce_meta.dtype}});

  std::vector<synTensor> bce_bwd_inputs = {self.syn_t, target.syn_t};

  if (is_weights_used)
    bce_bwd_inputs.push_back(weights->syn_t);

  bce_bwd_inputs.push_back(neg_grad[0].get());

  auto bce_bwd = BuildOp(
      graph,
      get_guid_with_precision("binary_cross_entropy_bwd", bce_meta.dtype),
      std::move(bce_bwd_inputs),
      {{bce_meta.shape, bce_meta.dtype, 0}},
      params.get(),
      size);

  // output of bce_bwd is the output of this op
  syn_out(0) = std::move(bce_bwd[0]);
}
} // namespace habana
