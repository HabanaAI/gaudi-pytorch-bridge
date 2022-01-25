/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <utility>

#include "generated/hpu_op.h"

namespace habana {

sizes_vec SmoothL1LossOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();
  if (reduction == at::Reduction::Reduction::None) {
    return {self.sizes().vec()};
  }
  return {{}};
}

sizes_vec SmoothL1LossBackwardOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 1);
  return {self.sizes().vec()};
}

void SmoothL1LossBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 1).sizes();

  float beta = stack.at(4).toScalar().to<float>();
  TORCH_CHECK(
      beta >= 0,
      "smooth_l1_loss_backward does not support negative values for beta.")
  auto mode = stack.at(3).toInt();
  float norm_factor = (mode == at::Reduction::Reduction::Mean)
      ? 1 / static_cast<float>(stack_tensor(stack, 1).numel())
      : 1;

  std::vector<synapse_helpers::tensor> t_l0;

  auto t_diff = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(1), syn_in(2)},
      {{inputshape, ScalarType()}});

  if (mode == at::Reduction::Reduction::Mean) {
    auto t_norm_factor =
        ConstantHelper(graph, norm_factor, ScalarType(), inputshape);

    auto t_mul = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), t_norm_factor.get()},
        {{inputshape, ScalarType()}});

    auto t_sign = BuildOp(
        graph,
        "sign_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {t_diff.at(0).get()},
        {{inputshape, ScalarType()}});

    if (beta == 0) {
      t_l0 = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
          {t_mul.at(0).get(), t_sign.at(0).get()},
          {{inputshape, ScalarType(), 0}});

      syn_out(0) = std::move(t_l0.at(0));
      return;
    }

    t_l0 = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {t_mul.at(0).get(), t_sign.at(0).get()},
        {{inputshape, ScalarType()}});

  } else {
    auto t_sign = BuildOp(
        graph,
        "sign_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {t_diff.at(0).get()},
        {{inputshape, ScalarType()}});

    if (beta == 0) {
      t_l0 = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
          {syn_in(0), t_sign.at(0).get()},
          {{inputshape, ScalarType(), 0}});

      syn_out(0) = std::move(t_l0.at(0));
      return;
    }

    t_l0 = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), t_sign.at(0).get()},
        {{inputshape, ScalarType()}});
  }

  auto t_mulfactor =
      ConstantHelper(graph, norm_factor / beta, ScalarType(), inputshape);

  auto t_l2_temp = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), t_mulfactor.get()},
      {{inputshape, ScalarType()}});

  auto t_l2 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_diff.at(0).get(), t_l2_temp.at(0).get()},
      {{inputshape, ScalarType()}});

  auto t_mask_const = ConstantHelper(graph, beta, ScalarType(), inputshape);

  auto t_abs = BuildOp(
      graph,
      "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_diff.at(0).get()},
      {{inputshape, ScalarType()}});

  auto mask = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_abs.at(0).get(), t_mask_const.get()},
      {{inputshape, ScalarType()}});

  auto grad_in = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask.at(0).get(), t_l2.at(0).get(), t_l0.at(0).get()},
      {{inputshape, ScalarType(), 0}});

  syn_out(0) = std::move(grad_in.at(0));
  return;
}

void SmoothL1LossOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& inputshape = stack_tensor(stack, 0).sizes();
  auto mode = stack.at(2).toInt();
  float beta = stack.at(3).toScalar().to<float>();
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")

  auto sub = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(1)},
      {{inputshape, ScalarType()}});

  if (beta == 0) {
    if (mode == at::Reduction::Reduction::None) {
      auto t_absdiff = BuildOp(
          graph,
          "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
          {sub.at(0).get()},
          {{inputshape, ScalarType(), 0}});

      syn_out(0) = std::move(t_absdiff.at(0));
      return;
    }

    auto t_absdiff = BuildOp(
        graph,
        "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {sub.at(0).get()},
        {{inputshape, ScalarType()}});

    auto n_dims = stack_tensor(stack, 0).dim();
    std::vector<synTensor> reduction_inputs = {t_absdiff[0].get()};
    std::vector<synapse_helpers::tensor> t_absdiff_flat;

    if (n_dims > 1) {
      auto reshape_outshape = stack_tensor(stack, 0).numel();
      t_absdiff_flat = BuildOp(
          graph,
          "reshape",
          {t_absdiff.at(0).get()},
          {{reshape_outshape, ScalarType()}});
      reduction_inputs = {t_absdiff_flat[0].get()};
    }

    std::string reduction_guid =
        (mode == at::Reduction::Mean) ? "reduce_mean_fwd_" : "reduce_sum_fwd_";
    ns_Reduction::Params node_params{};
    node_params.reductionDimension = 0;
    auto t_out = BuildOp(
        graph,
        reduction_guid + habana_helpers::name_suffix_from_type(ScalarType()),
        reduction_inputs,
        {{1, ScalarType(), 0}},
        &node_params,
        sizeof(node_params));

    syn_out(0) = std::move(t_out.at(0));
    return;
  }

  auto t_absdiff = BuildOp(
      graph,
      "abs_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {sub.at(0).get()},
      {{inputshape, ScalarType()}});

  auto t_beta = ConstantHelper(graph, beta, ScalarType(), inputshape);

  auto mask = BuildOp(
      graph,
      "less_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_absdiff.at(0).get(), t_beta.get()},
      {{inputshape, ScalarType()}});

  auto loss_params = std::make_shared<ns_MSELossKernel::Params>();
  loss_params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_NONE;
  auto t_mse = BuildOp(
      graph,
      "mse_loss_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(1)},
      {{inputshape, ScalarType()}},
      loss_params.get(),
      sizeof(ns_MSELossKernel::Params));

  auto t_mse_scale =
      ConstantHelper(graph, 0.5 / beta, ScalarType(), inputshape);

  auto t_l2 = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_mse.at(0).get(), t_mse_scale.get()},
      {{inputshape, ScalarType()}});

  auto t_b = ConstantHelper(graph, 0.5 * beta, ScalarType(), inputshape);

  auto t_l1 = BuildOp(
      graph,
      "sub_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {t_absdiff.at(0).get(), t_b.get()},
      {{inputshape, ScalarType()}});

  if (mode == at::Reduction::Reduction::None) {
    auto t_wh = BuildOp(
        graph,
        "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {mask.at(0).get(), t_l2.at(0).get(), t_l1.at(0).get()},
        {{inputshape, ScalarType(), 0}});

    syn_out(0) = std::move(t_wh.at(0));
    return;
  }

  auto t_wh = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask.at(0).get(), t_l2.at(0).get(), t_l1.at(0).get()},
      {{inputshape, ScalarType()}});

  auto n_dims = stack_tensor(stack, 0).dim();
  std::vector<synTensor> reduction_inputs = {t_wh[0].get()};
  std::vector<synapse_helpers::tensor> t_wh_flat;

  if (n_dims > 1) {
    auto reshape_outshape = stack_tensor(stack, 0).numel();
    t_wh_flat = BuildOp(
        graph,
        "reshape",
        {t_wh.at(0).get()},
        {{reshape_outshape, ScalarType()}});
    reduction_inputs = {t_wh_flat[0].get()};
  }

  std::string reduction_guid =
      (mode == at::Reduction::Mean) ? "reduce_mean_fwd_" : "reduce_sum_fwd_";
  ns_Reduction::Params node_params{};
  node_params.reductionDimension = 0;
  auto t_out = BuildOp(
      graph,
      reduction_guid + habana_helpers::name_suffix_from_type(ScalarType()),
      reduction_inputs,
      {{1, ScalarType(), 0}},
      &node_params,
      sizeof(node_params));

  syn_out(0) = std::move(t_out.at(0));
  return;
}
} // namespace habana
