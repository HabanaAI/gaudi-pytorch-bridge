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

namespace habana {

// Copied from habana_kernels/threshold_kernels.cpp
void ThresholdBackwardHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  // TODO: Remove this base class once [SW-65399] is resolved
  TORCH_CHECK(
      stack.size() == 3 || stack.size() == 4,
      "Incorrect size of inputs expected for threshold operator");

  TORCH_CHECK(stack[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(stack[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(stack[2].isScalar(), "Input arg3 type expected to be scalar");
  if (stack.size() == 4) {
    TORCH_CHECK(stack[3].isTensor(), "Input arg4 type expected to be tensor");
  }
  auto threshold = stack[2].toScalar();

  TORCH_CHECK(
      threshold.to<float>() == 0.0,
      "Threshold values other than 0 are not supported")

  OpBackend::AddNode(graph, stack, is_output_persistent_list);
}

void Threshold::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const float threshold = stack.at(1).toScalar().toFloat();
  const float value = stack.at(2).toScalar().toFloat();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  if (threshold == 0 && value == 0) {
    // if threshold and value are zero then the op is same perform same as relu
    auto output = BuildOp(
        graph,
        "relu_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{outshape, ScalarType(), is_output_persistent_list[0], true}});
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
        {{outshape, ScalarType(), is_output_persistent_list[0], true}});

    syn_out(0) = std::move(output[0]);
  }
}
} // namespace habana