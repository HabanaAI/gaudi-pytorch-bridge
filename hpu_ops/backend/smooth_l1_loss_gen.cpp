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

OutputMetaDataVector SmoothL1LossMeta(const at::Stack& stack) {
  float beta = stack.at(3).toScalar().to<float>();
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = (reduction == at::Reduction::Reduction::None)
      ? self.sizes().vec()
      : std::vector<int64_t>{};
  return {meta};
}

OutputMetaDataVector SmoothL1LossBackwardMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return {meta};
}

void SmoothL1LossBwdOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = SmoothL1LossBackwardMeta(stack)[0];

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
      get_guid_with_precision("sub", meta.dtype),
      {syn_in(1), syn_in(2)},
      {{meta.shape, meta.dtype}});

  if (mode == at::Reduction::Reduction::Mean) {
    auto t_norm_factor =
        ConstantHelper(graph, norm_factor, meta.dtype, meta.shape);

    auto t_mul = BuildOp(
        graph,
        get_guid_with_precision("mult", meta.dtype),
        {syn_in(0), t_norm_factor.get()},
        {{meta.shape, meta.dtype}});

    auto t_sign = BuildOp(
        graph,
        get_guid_with_precision("sign_fwd", meta.dtype),
        {t_diff.at(0).get()},
        {{meta.shape, meta.dtype}});

    if (beta == 0) {
      t_l0 = BuildOp(
          graph,
          get_guid_with_precision("mult", meta.dtype),
          {t_mul.at(0).get(), t_sign.at(0).get()},
          {{meta.shape, meta.dtype, 0}});

      syn_out(0) = std::move(t_l0.at(0));
      return;
    }

    t_l0 = BuildOp(
        graph,
        get_guid_with_precision("mult", meta.dtype),
        {t_mul.at(0).get(), t_sign.at(0).get()},
        {{meta.shape, meta.dtype}});

  } else {
    auto t_sign = BuildOp(
        graph,
        get_guid_with_precision("sign_fwd", meta.dtype),
        {t_diff.at(0).get()},
        {{meta.shape, meta.dtype}});

    if (beta == 0) {
      t_l0 = BuildOp(
          graph,
          get_guid_with_precision("mult", meta.dtype),
          {syn_in(0), t_sign.at(0).get()},
          {{meta.shape, meta.dtype, 0}});

      syn_out(0) = std::move(t_l0.at(0));
      return;
    }

    t_l0 = BuildOp(
        graph,
        get_guid_with_precision("mult", meta.dtype),
        {syn_in(0), t_sign.at(0).get()},
        {{meta.shape, meta.dtype}});
  }

  auto t_mulfactor =
      ConstantHelper(graph, norm_factor / beta, meta.dtype, meta.shape);

  auto t_l2_temp = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {syn_in(0), t_mulfactor.get()},
      {{meta.shape, meta.dtype}});

  auto t_l2 = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {t_diff.at(0).get(), t_l2_temp.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto t_mask_const = ConstantHelper(graph, beta, meta.dtype, meta.shape);

  auto t_abs = BuildOp(
      graph,
      get_guid_with_precision("abs_fwd", meta.dtype),
      {t_diff.at(0).get()},
      {{meta.shape, meta.dtype}});

  auto mask = BuildOp(
      graph,
      get_guid_with_precision("less_fwd", meta.dtype),
      {t_abs.at(0).get(), t_mask_const.get()},
      {{meta.shape, at::kBool}});

  auto grad_in = BuildOp(
      graph,
      get_guid_with_precision("where_fwd", meta.dtype),
      {mask.at(0).get(), t_l2.at(0).get(), t_l0.at(0).get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(grad_in.at(0));
}
} // namespace habana
