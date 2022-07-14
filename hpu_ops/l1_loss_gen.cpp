/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/l1_loss.h"
#include "generated/l1_loss_backward.h"

namespace habana {

sizes_vec L1LossOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();
  if (reduction == at::Reduction::Reduction::None) {
    return {self.sizes().vec()};
  }
  return {{}};
}

sizes_vec L1LossBackwardOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 1);
  return {self.sizes().vec()};
}

void L1LossOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 0).sizes();
  auto mode = stack.at(2).toInt();

  auto sub = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(1)},
      {{inputshape, ScalarType()}});

  if (mode == at::Reduction::Reduction::None) {
    auto absdiff = BuildOp(
        graph,
        "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {sub.at(0).get()},
        {{inputshape, ScalarType(), 0}});

    syn_out(0) = std::move(absdiff.at(0));
  } else {
    auto absdiff = BuildOp(
        graph,
        "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {sub.at(0).get()},
        {{inputshape, ScalarType()}});
    auto reshape_outshape = stack_tensor(stack, 0).numel();

    // Falttening the input from abs_fwd Guid
    auto inp_flatten =
        ReshapeHelper(graph, absdiff[0].get(), reshape_outshape, ScalarType());

    std::string reduction_guid =
        (mode == at::Reduction::Mean) ? "reduce_mean_fwd_" : "reduce_sum_fwd_";

    size_t size = 0;
    PARAMS_STUB(ns_Reduction::Params);

    // passing input to reduce_mean or reduce_sum guid
    auto t_out = BuildOp(
        graph,
        reduction_guid + habana_helpers::name_suffix_from_type(ScalarType()),
        {inp_flatten.get()},
        {{1, ScalarType(), 0}},
        params.get(),
        size);

    syn_out(0) = std::move(t_out.at(0));
  }
}

void L1LossBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 1).sizes();
  auto reduction = stack.at(3).toInt();

  double norm_factor = (reduction == at::Reduction::Reduction::Mean)
      ? 1 / static_cast<double>(stack_tensor(stack, 1).numel())
      : 1;
  auto t_norm_factor =
      ConstantHelper(graph, norm_factor, ScalarType(), inputshape);
  auto t_diff = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), syn_in(2)},
      {{inputshape, ScalarType()}});
  // Computes output, If t_diff > 0, output = 1.
  // if t_diff < 0, output = -1.
  // if t_diff == 0, output = 0.
  auto t_sign = BuildOp(
      graph,
      "sign_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_diff.at(0).get()},
      {{inputshape, ScalarType()}});

  auto t_mul = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), t_norm_factor.get()},
      {{inputshape, ScalarType()}});
  auto grad_in = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_mul.at(0).get(), t_sign.at(0).get()},
      {{inputshape, ScalarType(), 0}});
  syn_out(0) = std::move(grad_in.at(0));
}
} // namespace habana
