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

FALLBACK_CHECK(
    threshold_backward_fallback,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold) {
  static_cast<void>(grad_output);
  static_cast<void>(self);
  // Threshold values other than 0 are not supported
  return threshold.toDouble() == 0;
};

FALLBACK_CHECK(
    threshold_backward_out_fallback,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold,
    const at::Tensor& grad_input) {
  static_cast<void>(grad_output);
  static_cast<void>(self);
  static_cast<void>(grad_input);
  // Threshold values other than 0 are not supported
  return threshold.toDouble() == 0;
};

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
} // namespace habana
