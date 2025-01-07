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

#include "generated/backend/embedding.h"
#include "generated/backend/embedding_dense_backward.h"

namespace habana {
OutputMetaDataVector EmbeddingMeta(const at::Stack& stack) {
  const auto& weight = stack_tensor(stack, 0);
  const auto& indices = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = weight.scalar_type();
  if (indices.dim() == 1) {
    meta.shape = weight.sizes().vec();
    meta.shape[0] = indices.numel();
  } else {
    meta.shape = indices.sizes().vec();
    for (int64_t d : weight.sizes().slice(1)) {
      meta.shape.push_back(d);
    }
  }
  return {meta};
}

std::shared_ptr<void> FillEmbeddingRenormFwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_EmbeddingRenormFwdKernel::Params);
  params->max_norm = stack.at(2).toScalar().to<double>();
  params->norm_type = stack.at(3).toScalar().to<double>();
  return params;
}

} // namespace habana
