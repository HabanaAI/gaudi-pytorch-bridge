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
#include "generated/backend/glu.h"
#include "generated/backend/glu_backward.h"
#include "pytorch_helpers/habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

OutputMetaDataVector GluMeta(const at::Stack& stack) {
  OutputMetaData meta;
  auto self = stack.at(0).toTensor();
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();

  const int64_t axis = stack.at(1).toInt();
  auto dim = static_cast<size_t>(
      (axis >= 0) ? axis : stack.at(0).toTensor().dim() + axis);
  meta.shape[dim] = meta.shape[dim] / 2;
  return {meta};
}

OutputMetaDataVector GluBwdMeta(const at::Stack& stack) {
  OutputMetaData meta;
  auto self = stack.at(1).toTensor();
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

FillParamsT FillGluParams(const at::Stack& stack, const size_t dim_index) {
  const auto self = stack_tensor(stack, dim_index - 1);
  const int64_t dim = at::maybe_wrap_dim(
      stack.at(dim_index).toInt(), self.dim(), /*wrap_scalar=*/true);
  PARAMS_STUB(ns_GatherKernel::Params);
  params->axis = safe_convert<int>(dim, "axis"sv);
  return paramsT;
}
FillParamsT FillGluFwdParams(const at::Stack& stack) {
  return FillGluParams(stack, 1U /*dim_index FWD*/);
}

FillParamsT FillGluBwdParams(const at::Stack& stack) {
  return FillGluParams(stack, 2U /*dim_index BWD*/);
}

} // namespace habana
