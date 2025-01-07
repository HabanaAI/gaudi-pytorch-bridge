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
#include "generated/backend/clone.h"

namespace habana {

OutputMetaDataVector CloneMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  OutputMetaData meta;

  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  if (stack.at(1).isNone()) {
    meta.mem_format = self.suggest_memory_format();
  } else {
    meta.mem_format = stack.at(1).toMemoryFormat();
  }

  return {meta};
}

} // namespace habana
