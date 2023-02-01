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
#include "generated/backend/threshold.h"
#include "generated/backend/threshold_backward.h"

namespace habana {

void Threshold::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const float threshold = stack.at(1).toScalar().toFloat();
  const float value = stack.at(2).toScalar().toFloat();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  if (threshold == 0 && value == 0) {
    // if threshold and value are zero then the op is same perform same as relu
    auto output = BuildOp(
        graph,
        "relu_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(output[0]);
  } else {
    auto threshold_gen = ConstantHelper(graph, threshold, ScalarType());
    const at::ScalarType& result_type = c10::ScalarType::Bool;
    // Checking wheather innput is greater than threshold returns boolean tensor
    auto greaterthan_threshold = BuildOp(
        graph,
        "greater_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), threshold_gen.get()},
        {{outshape, result_type}});
    auto value_gen = ConstantHelper(graph, value, ScalarType());
    // returns input if it is true else return the value
    auto output = BuildOp(
        graph,
        "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {greaterthan_threshold[0].get(), syn_in(0), value_gen.get()},
        {{outshape, ScalarType(), 0}});

    syn_out(0) = std::move(output[0]);
  }
}

void ThresholdBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  TORCH_CHECK(
      stack.size() == 3,
      "Incorrect size of inputs expected for threshold operator");

  TORCH_CHECK(stack[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(stack[1].isTensor(), "Input arg2 type expected to be tensor");

  auto grad_output = stack[0].toTensor();
  auto self = stack[1].toTensor();
  TORCH_CHECK(grad_output.sizes() == self.sizes(), "Input sizes must be equal");

  return OpBackend::AddNode(graph, stack);
}

} // namespace habana
