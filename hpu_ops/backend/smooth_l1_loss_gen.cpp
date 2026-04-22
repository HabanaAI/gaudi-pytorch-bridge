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

#include "generated/backend/smooth_l1_loss.h"
#include "generated/backend/smooth_l1_loss_backward.h"

namespace habana {

FillParamsT FillSmoothL1LossParams(const at::Stack& stack, const int offset) {
  PARAMS_STUB(ns_SmoothL1LossKernel::Params);
  auto mode = stack.at(offset + 2).toInt();
  if (mode == at::Reduction::Reduction::Mean) {
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_MEAN;
  } else if (mode == at::Reduction::Reduction::Sum) {
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_SUM;
  } else {
    params->mode = LossMode_t::LOSS_REDUCTION_MODE_NONE;
  }
  params->beta = stack.at(offset + 3).toScalar().to<float>();
  return paramsT;
}

FillParamsT FillSmoothL1LossFwdParams(const at::Stack& stack) {
  return FillSmoothL1LossParams(stack, 0);
}

FillParamsT FillSmoothL1LossBwdParams(const at::Stack& stack) {
  return FillSmoothL1LossParams(stack, 1);
}

OutputMetaDataVector SmoothL1LossMeta(const at::Stack& stack) {
  float beta = stack.at(3).toScalar().to<float>();
  HABANA_ASSERT(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = self.scalar_type();
  meta.shape = (reduction == at::Reduction::Reduction::None)
      ? self.sizes().vec()
      : std::vector<int64_t>{};
  return metaVec;
}

OutputMetaDataVector SmoothL1LossBackwardMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return metaVec;
}

} // namespace habana
