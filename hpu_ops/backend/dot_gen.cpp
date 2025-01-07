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

#include "generated/backend/dot.h"

namespace habana {
sizes_vec DotOutputShape(const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other = stack_tensor(stack, 1);
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "Dot Op: 1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  TORCH_CHECK(
      self.sizes() == other.sizes(),
      "Dot Op: Tensor must have same size, but got ",
      self.sizes(),
      "and ",
      other.sizes(),
      "size tensors");
  return {{}};
}

OutputMetaDataVector DotMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = DotOutputShape(stack)[0];
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      c10::nullopt,
      false,
      false);

  return {meta};
}
} // namespace habana
