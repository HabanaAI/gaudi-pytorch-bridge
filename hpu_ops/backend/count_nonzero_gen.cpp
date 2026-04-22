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
#include "generated/backend/count_nonzero.h"

namespace habana {

static std::vector<int64_t> get_dims_from_stack(const at::Stack& stack) {
  auto tensor_rank =
      static_cast<int64_t>(stack_tensor(stack, 0).sizes().size());

  std::vector<int64_t> dims;
  if (stack[1].isIntList()) {
    dims = stack[1].toIntList().vec();
  } else if (stack[1].isInt()) {
    dims = {stack[1].toInt()};
  }
  for (int64_t& dim : dims) {
    if (dim < 0) {
      dim = tensor_rank + dim;
    }
  }

  return dims;
}

FillParamsT FillCountNonzeroParams(const at::Stack& stack) {
  PARAMS_STUB(ns_CountNonZero::Params);
  params->dims = 0;

  std::vector<int64_t> dims = get_dims_from_stack(stack);
  if (dims.empty()) {
    params->dims = (1 << stack_tensor(stack, 0).sizes().size()) - 1;
  } else {
    for (auto dim : dims) {
      params->dims |= (1 << dim);
    }
  }

  return paramsT;
}

OutputMetaDataVector CountNonzeroMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto self_shape = self.sizes();

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();

  meta.dtype = c10::ScalarType::Long;

  std::vector<int64_t> dims = get_dims_from_stack(stack);
  meta.shape = {};
  meta.shape.reserve(self_shape.size());

  if (!dims.empty()) {
    for (uint64_t i = 0; i < self_shape.size(); ++i) {
      if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
        meta.shape.push_back(self_shape[i]);
      }
    }
  }

  return metaVec;
}

} // namespace habana
