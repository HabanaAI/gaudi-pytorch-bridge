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

#include "generated/backend/mamba_pscan.h"

namespace habana {

OutputMetaDataVector MambaPscanMeta(const at::Stack& stack) {
  const auto& state = stack.at(0).toTensor();
  const auto& in_x = stack.at(1).toTensor();
  OutputMetaData meta;
  meta.dtype = state.scalar_type();
  meta.shape = in_x.sizes().vec();

  if (in_x.dim() < 2 || state.dim() < 2)
    TORCH_CHECK_INDEX(false, "in_x and state dim cannot be less than 2.");

  meta.shape[in_x.dim() - 2] = state.sizes().vec()[state.dim() - 2];

  return {meta};
}
} // namespace habana
