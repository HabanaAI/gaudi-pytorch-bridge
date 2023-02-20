/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/huber_loss.h"
#include "generated/backend/huber_loss_backward.h"

namespace habana {

std::shared_ptr<void> FillHuberLossFwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_HuberLossKernel::Params);

  double delta = stack.at(3).toScalar().to<double>();
  params->delta = delta;

  auto mode = stack.at(2).toInt();
  if (mode == at::Reduction::Reduction::Mean)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  else if (mode == at::Reduction::Reduction::Sum)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
  else
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
  return params;
}

sizes_vec HuberLossOutputShape(const at::Stack& stack) {
  double delta = stack.at(3).toScalar().to<double>();
  TORCH_CHECK(
      delta >= 0, "huber_loss does not support negative values for delta.")
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
} // namespace habana
