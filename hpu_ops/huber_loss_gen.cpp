/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/huber_loss.h"
#include "generated/huber_loss_backward.h"

namespace habana {

sizes_vec HuberLossOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();
  if (reduction == at::Reduction::Reduction::None) {
    return {self.sizes().vec()};
  }
  return {{}};
}

sizes_vec HuberLossBackwardOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);
  return {self.sizes().vec()};
}

void HuberLossBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 1).sizes();

  float delta = stack.at(4).toScalar().to<float>();
  TORCH_CHECK(
      delta >= 0,
      "huber_loss_backward does not support negative values for delta.")
  auto mode = stack.at(3).toInt();
  float norm_factor = (mode == at::Reduction::Reduction::Mean)
      ? 1 / static_cast<float>(stack_tensor(stack, 1).numel())
      : 1;

  auto norm = ConstantHelper(graph, norm_factor, ScalarType(), inputshape);

  auto delta_const = ConstantHelper(graph, delta, ScalarType(), inputshape);

  auto t_diff = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), syn_in(2)},
      {{inputshape, ScalarType()}});

  auto t_mul = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), norm.get()},
      {{inputshape, ScalarType()}});

  auto t_sign = BuildOp(
      graph,
      "sign_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_diff.at(0).get()},
      {{inputshape, ScalarType()}});

  auto t_0 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_mul.at(0).get(), delta_const.get()},
      {{inputshape, ScalarType()}});

  auto t_1 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_0.at(0).get(), t_sign.at(0).get()},
      {{inputshape, ScalarType()}});

  auto t_2 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_diff.at(0).get(), t_mul.at(0).get()},
      {{inputshape, ScalarType()}});

  auto t_abs = BuildOp(
      graph,
      "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_diff.at(0).get()},
      {{inputshape, ScalarType()}});

  auto mask_bwd = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_abs.at(0).get(), delta_const.get()},
      {{inputshape, at::kBool}});

  auto grad_in = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask_bwd.at(0).get(), t_2.at(0).get(), t_1.at(0).get()},
      {{inputshape, ScalarType(), 0}});

  syn_out(0) = std::move(grad_in.at(0));
  return;
}

void HuberLossOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 0).sizes();
  auto mode = stack.at(2).toInt();
  double delta = stack.at(3).toScalar().to<double>();
  TORCH_CHECK(
      delta >= 0, "huber_loss does not support negative values for delta.")

  auto sub = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(1)},
      {{inputshape, ScalarType()}});

  auto abs = BuildOp(
      graph,
      "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {sub.at(0).get()},
      {{inputshape, ScalarType()}});

  auto delta_const = ConstantHelper(graph, delta, ScalarType(), inputshape);

  auto const_05 = ConstantHelper(graph, 0.5, ScalarType(), inputshape);

  auto mask = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {abs.at(0).get(), delta_const.get()},
      {{inputshape, at::kBool}});

  auto sq_out = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {abs.at(0).get(), abs.at(0).get()},
      {{inputshape, ScalarType()}});

  auto result_true = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {sq_out.at(0).get(), const_05.get()},
      {{inputshape, ScalarType()}});

  auto mul_out = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {const_05.get(), delta_const.get()},
      {{inputshape, ScalarType()}});

  auto sub_out = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {abs.at(0).get(), mul_out.at(0).get()},
      {{inputshape, ScalarType()}});

  auto result_false = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {sub_out.at(0).get(), delta_const.get()},
      {{inputshape, ScalarType()}});

  if (mode == at::Reduction::Reduction::None) {
    auto condition_out = BuildOp(
        graph,
        "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {mask.at(0).get(), result_true.at(0).get(), result_false.at(0).get()},
        {{inputshape, ScalarType(), 0}});

    syn_out(0) = std::move(condition_out.at(0));
    return;
  }

  auto condition_out = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask.at(0).get(), result_true.at(0).get(), result_false.at(0).get()},
      {{inputshape, ScalarType()}});

  auto n_dims = stack_tensor(stack, 0).dim();
  std::vector<synTensor> reduction_inputs = {condition_out[0].get()};
  std::vector<synapse_helpers::tensor> condition_out_flat;

  if (n_dims > 1) {
    auto reshape_outshape = stack_tensor(stack, 0).numel();
    condition_out_flat.emplace_back(BuildReshape(
        this,
        graph,
        condition_out.at(0).get(),
        reshape_outshape,
        ScalarType()));
    reduction_inputs = {condition_out_flat[0].get()};
  }

  std::string reduction_guid =
      (mode == at::Reduction::Mean) ? "reduce_mean_fwd_" : "reduce_sum_fwd_";
  ns_Reduction::Params node_params{};
  node_params.reductionDimension = 0;
  auto output = BuildOp(
      graph,
      reduction_guid + habana_helpers::name_suffix_from_type(ScalarType()),
      reduction_inputs,
      {{1, ScalarType(), 0}},
      &node_params,
      sizeof(node_params));

  syn_out(0) = std::move(output.at(0));
  return;
}
} // namespace habana
