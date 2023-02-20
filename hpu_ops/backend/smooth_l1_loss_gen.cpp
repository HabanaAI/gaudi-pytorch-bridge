/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/smooth_l1_loss.h"
#include "generated/backend/smooth_l1_loss_backward.h"

namespace habana {

std::shared_ptr<void> FillSmoothL1LossFwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SmoothL1LossKernel::Params);
  auto mode = stack.at(2).toInt();
  if (mode == at::Reduction::Reduction::Mean)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  else if (mode == at::Reduction::Reduction::Sum)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
  else
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
  params->beta = stack.at(3).toScalar().to<float>();
  return params;
}

sizes_vec SmoothL1LossOutputShape(const at::Stack& stack) {
  float beta = stack.at(3).toScalar().to<float>();
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();
  if (reduction == at::Reduction::Reduction::None) {
    return {self.sizes().vec()};
  }
  return {{}};
}

sizes_vec SmoothL1LossBackwardOutputShape(const at::Stack& stack) {
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
      {{inputshape, at::kBool}});

  auto grad_in = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {mask.at(0).get(), t_l2.at(0).get(), t_l0.at(0).get()},
      {{inputshape, ScalarType(), 0}});

  syn_out(0) = std::move(grad_in.at(0));
  return;
}
} // namespace habana
