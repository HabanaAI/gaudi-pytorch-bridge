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
constexpr int64_t index_of_fwd_weight_tensor = 2;
constexpr int64_t index_of_fwd_mode = 3;

namespace habana {

sizes_vec BinaryCrossEntropyFwdOutputShape(const at::Stack& stack, bool) {
  constexpr int64_t index_of_self = 0;
  auto reduction = stack.at(index_of_fwd_mode).toInt();
  if (reduction == at::Reduction::Reduction::None)
    return {stack.at(index_of_self).toTensor().sizes().vec()};
  return {{}};
}

sizes_vec BinaryCrossEntropyBwdOutputShape(const at::Stack& stack, bool) {
  constexpr int64_t index_of_self = 1;
  return {stack.at(index_of_self).toTensor().sizes().vec()};
}

static std::shared_ptr<void> BceParams(
    const at::Stack& stack,
    size_t& size,
    bool is_backward) {
  PARAMS_STUB(ns_BinaryCrossEntropy::ParamsOptionalSigmoid);
  constexpr int64_t index_of_bwd_mode = 4;
  constexpr int64_t index_of_bwd_weight_tensor = 3;

  auto mode =
      stack.at((is_backward) ? index_of_bwd_mode : index_of_fwd_mode).toInt();

  auto weight = false; // By default, weight is set to false

  if (is_backward && (!stack.at(index_of_bwd_weight_tensor).isNone()))
    weight = is_backward; // Set to true for Backward variant
  else if (!stack.at(index_of_fwd_weight_tensor).isNone())
    weight = !is_backward; // Set to true for Forward variant

  params->isWeightsUsed = weight;
  params->binaryCrossEntropyWithoutSigmoid = true;

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
  constexpr int64_t index_of_self = 0;
  constexpr int64_t index_of_target = 1;

  auto bce_output_shape = BinaryCrossEntropyFwdOutputShape(stack)[0];

  size_t size = 0;
  bool is_backward = false;
  const auto& params = BceParams(stack, size, is_backward);

  auto target = stack.at(index_of_target).toTensor();
  std::vector<int64_t> target_shape = target.sizes().vec();

  if (!stack.at(index_of_fwd_weight_tensor).isNone()) {
    // Added to provide broadcast support for weight tensors
    auto broadcast_weight = BuildOp(
        graph,
        "broadcast_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(index_of_fwd_weight_tensor)},
        {{target_shape, ScalarType()}});

    auto bce_fwd = BuildOp(
        graph,
        "binary_cross_entropy_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(index_of_self),
         syn_in(index_of_target),
         broadcast_weight[0].get()},
        {{bce_output_shape, ScalarType(), 0}},
        params.get(),
        size);

    // output of bce_fwd is the output of this op
    syn_out(0) = std::move(bce_fwd[0]);
  } else {
    auto bce_fwd = BuildOp(
        graph,
        "binary_cross_entropy_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(index_of_self), syn_in(index_of_target)},
        {{bce_output_shape, ScalarType(), 0}},
        params.get(),
        size);

    // output of bce_fwd is the output of this op
    syn_out(0) = std::move(bce_fwd[0]);
  }
}

// Backward variant
void BinaryCrossEntropyBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr int64_t index_of_grad = 0;
  constexpr int64_t index_of_self = 1;
  constexpr int64_t index_of_target = 2;

  auto bce_output_shape = BinaryCrossEntropyBwdOutputShape(stack)[0];
  size_t size = 0;
  bool is_backward = true;

  const auto& params = BceParams(stack, size, is_backward);

  auto neg_grad = BuildOp(
      graph,
      "neg_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(index_of_grad)},
      {{{1}, ScalarType()}});

  auto bce_bwd = BuildOp(
      graph,
      "binary_cross_entropy_bwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(index_of_self), syn_in(index_of_target), neg_grad[0].get()},
      {{bce_output_shape, ScalarType(), 0}},
      params.get(),
      size);

  // output of bce_bwd is the output of this op
  syn_out(0) = std::move(bce_bwd[0]);
}
} // namespace habana