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

#include "generated/backend/take.h"

namespace habana {

OutputMetaDataVector TakeMeta(const at::Stack& stack) {
  const auto input = stack_tensor(stack, 0);
  const auto index = stack_tensor(stack, 1);

  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = index.sizes().vec();
  return {meta};
}
} // namespace habana
