/**
 * Copyright (c) 2026 Intel Corporation
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

#include "generated/backend/l2_norm.h"

namespace habana {

FillParamsT FillL2NormParams(const at::Stack& stack) {
  const auto epsilon = stack.at(1).toScalar().toFloat();
  PARAMS_STUB(ns_L2normKernel::Params);

  params->epsilon = epsilon;

  return paramsT;
}

OutputMetaDataVector L2NormMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.dtype = at::kFloat;
  meta.shape = input.sizes().vec();
  return {meta};
}

SharedMetaDataVector L2NormSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& input = stack.at(0).toTensor();

  SharedMetaData sharedMeta("l2_norm_fwd");
  sharedMeta.inputs_data.emplace_back(getSharedMetaFromTensor(input));
  sharedMeta.outputs_data.emplace_back(input.dim(), at::kFloat);
  return {sharedMeta};
}

} // namespace habana
