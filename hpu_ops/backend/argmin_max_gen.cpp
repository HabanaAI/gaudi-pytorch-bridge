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

#include "common/utils.h"
#include "generated/backend/argmax.h"
#include "generated/backend/argmin.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

OutputMetaDataVector ArgMinMaxMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const auto dimOpt = stack.at(1);
  const bool keepdim = stack.at(2).toBool();

  auto dimVector = dimOpt.isNone() ? std::vector<int64_t>{}
                                   : std::vector<int64_t>{dimOpt.toInt()};

  OutputMetaData meta;
  meta.shape = ReductionOutputShape(self, dimVector, keepdim)[0];
  meta.dtype = c10::ScalarType::Long;
  return {meta};
}

std::shared_ptr<void> FillArgMinMaxParams(
    const at::Stack& stack,
    size_t& size) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dimOpt = stack.at(1).toOptional<int64_t>();

  PARAMS_STUB(ns_Reduction::ParamsV2);
  params->reductionDimensionMask = ReductionMask(self, dimOpt);
  params->keepDim = stack.at(2).toBool();
  return params;
}

} // namespace habana
