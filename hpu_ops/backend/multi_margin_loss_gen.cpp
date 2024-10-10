/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/multi_margin_loss.h"

namespace habana {

static std::shared_ptr<void> MultiMarginLossParamsCommon(
    size_t& size,
    int p,
    float margin,
    int64_t reduction) {
  PARAMS_STUB(ns_MultiMarginLoss::Params);
  params->p = p;
  params->margin = margin;
  switch (reduction) {
    case at::Reduction::Reduction::None:
      params->mode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported reduction mode in multi_margin_loss: ",
          reduction);
  }
  return params;
}

std::shared_ptr<void> FillMultiMarginLossParams(
    const at::Stack& stack,
    size_t& size) {
  int p = stack.at(2).toInt();
  float margin = stack.at(3).toScalar().toDouble();
  int64_t reduction = stack.at(5).toInt();

  return MultiMarginLossParamsCommon(size, p, margin, reduction);
}

std::shared_ptr<void> FillMultiMarginLossBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  int p = stack.at(3).toInt();
  float margin = stack.at(4).toScalar().toDouble();
  int64_t reduction = stack.at(6).toInt();

  return MultiMarginLossParamsCommon(size, p, margin, reduction);
}

OutputMetaDataVector MultiMarginLossMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& target = stack.at(1).toTensor();
  at::ScalarType dtype = input.scalar_type();

  int64_t reduction = stack.at(5).toInt();

  std::vector<int64_t> shape = reduction == at::Reduction::None
      ? std::vector<int64_t>{target.sizes().back()}
      : std::vector<int64_t>{};

  return {OutputMetaData(dtype, shape)};
}

OutputMetaDataVector MultiMarginLossBackwardMeta(const at::Stack& stack) {
  const auto& input = stack.at(1).toTensor();

  return {OutputMetaData(input.scalar_type(), input.sizes().vec())};
}

} // namespace habana