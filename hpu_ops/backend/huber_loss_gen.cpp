/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "generated/backend/huber_loss.h"
#include "generated/backend/huber_loss_backward.h"

namespace habana {

FillParamsT FillHuberLossParams(const at::Stack& stack, const int offset) {
  PARAMS_STUB(ns_HuberLossKernel::Params);

  double delta = stack.at(offset + 3).toScalar().to<double>();
  params->delta = delta;

  auto mode = stack.at(offset + 2).toInt();
  if (mode == at::Reduction::Reduction::Mean)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  else if (mode == at::Reduction::Reduction::Sum)
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
  else
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
  return paramsT;
}

FillParamsT FillHuberLossFwdParams(const at::Stack& stack) {
  return FillHuberLossParams(stack, 0);
}

FillParamsT FillHuberLossBwdParams(const at::Stack& stack) {
  return FillHuberLossParams(stack, 1);
}

OutputMetaDataVector HuberLossMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();
  double delta = stack.at(3).toScalar().to<double>();
  HABANA_ASSERT(
      delta >= 0, "huber_loss does not support negative values for delta.")

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = {};
  if (reduction == at::Reduction::Reduction::None)
    meta.shape = self.sizes().vec();

  return {meta};
}

OutputMetaDataVector HuberLossBackwardMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

} // namespace habana
