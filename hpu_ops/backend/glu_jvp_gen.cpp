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

#include "generated/backend/glu_jvp.h"

namespace habana {

OutputMetaDataVector GluJvpMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& output = metaVec.front();
  const auto input_tensor = stack.at(0).toTensor();
  output.shape = input_tensor.sizes().vec();
  output.dtype = input_tensor.scalar_type();
  return metaVec;
}

FillParamsT FillGluJvpParams(const at::Stack& stack) {
  const auto dim = stack.at(3).toScalar().toInt();

  PARAMS_STUB(ns_GatherKernel::Params);
  params->axis = dim;
  return paramsT;
}
} // namespace habana
