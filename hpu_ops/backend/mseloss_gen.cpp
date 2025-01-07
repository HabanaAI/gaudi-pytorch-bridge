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

#include "generated/backend/mse_loss.h"

namespace habana {

OutputMetaDataVector MseLossFwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int64_t reduction = stack.at(2).toInt();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = (reduction == at::Reduction::Reduction::None)
      ? self.sizes().vec()
      : std::vector<int64_t>{};
  return {meta};
}

OutputMetaDataVector MseLossBwdMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return {meta};
}

std::shared_ptr<void> FillMseLossParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_MSELossKernel::Params);

  auto mode = stack.at(stack.at(2).isInt() ? 2 : 3).toInt();
  switch (mode) {
    case at::Reduction::Reduction::None:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_NONE;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_SUM;
      break;
    default:
      TORCH_CHECK(false, "Unsupported reduction mode in mseloss: ", mode);
  }
  return params;
}

} // namespace habana
