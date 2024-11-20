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

#include "generated/backend/signbit.h"

namespace habana {

OutputMetaDataVector SignbitMeta(const at::Stack& stack) {
  constexpr size_t SELF_TENSOR_INDEX = 0;

  OutputMetaData meta;
  const at::Tensor& self = stack_tensor(stack, SELF_TENSOR_INDEX);

  meta.shape = self.sizes().vec();
  meta.dtype = at::kBool;

  return {meta};
}

} // namespace habana
