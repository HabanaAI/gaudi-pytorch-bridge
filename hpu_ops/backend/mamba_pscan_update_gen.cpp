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

#include "generated/backend/mamba_pscan_update.h"

namespace habana {

OutputMetaDataVector MambaPscanUpdateMeta(const at::Stack& stack) {
  const auto& in_x = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.dtype = in_x.scalar_type();
  meta.shape = in_x.sizes().vec();


  return {meta};
}
} // namespace habana
