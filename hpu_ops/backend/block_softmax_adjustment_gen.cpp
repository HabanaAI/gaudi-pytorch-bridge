/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/block_softmax_adjustment.h"

namespace habana {

OutputMetaDataVector BlockSoftmaxAdjustmentMeta(const at::Stack& stack) {
  auto block_maxes = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.shape = block_maxes.sizes().vec();
  meta.dtype = block_maxes.scalar_type();

  return {meta};
}

FillParamsT BlockSoftmaxAdjustmentParams(const at::Stack& stack) {
  const auto batchSize = stack.at(3).toScalar().toInt();

  PARAMS_STUB(ns_BlockSoftmaxAdjustment::Params);
  params->batchSize = batchSize;
  return paramsT;
}

} // namespace habana
