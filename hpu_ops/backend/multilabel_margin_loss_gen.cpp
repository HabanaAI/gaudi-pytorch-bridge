/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "generated/backend/multilabel_margin_loss_forward.h"

namespace habana {

std::shared_ptr<void> FillMultilabelMarginLossParamsCommon(
    const at::Stack&,
    size_t& size,
    int64_t reduction) {
  PARAMS_STUB(ns_MultilabelMarginLoss::Params);
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
          "Unsupported reduction mode in multilabel_margin_loss: ",
          reduction);
  }
  return params;
}

std::shared_ptr<void> FillMultilabelMarginLossParams(
    const at::Stack& stack,
    size_t& size) {
  return FillMultilabelMarginLossParamsCommon(stack, size, stack.at(2).toInt());
}

std::shared_ptr<void> FillMultilabelMarginLossBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  return FillMultilabelMarginLossParamsCommon(stack, size, stack.at(3).toInt());
}

OutputMetaDataVector MultilabelMarginLossBackwardMeta(const at::Stack& stack) {
  const auto& input = stack.at(1).toTensor();

  return {OutputMetaData(input.scalar_type(), input.sizes().vec())};
}

OutputMetaDataVector MultilabelMarginLossMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& target = stack.at(1).toTensor();
  at::ScalarType dtype = input.scalar_type();

  int64_t reduction = stack.at(2).toInt();

  std::vector<int64_t> shape =
      reduction == at::Reduction::None && input.sizes().size() == 2
      ? std::vector<int64_t>{target.sizes().front()}
      : std::vector<int64_t>{};

  return {
      OutputMetaData(dtype, shape),
      OutputMetaData(torch::kLong, target.sizes().vec())};
}

} // namespace habana