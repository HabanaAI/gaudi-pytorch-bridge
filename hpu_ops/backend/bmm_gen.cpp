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

#include "generated/backend/bmm.h"

namespace habana {
OutputMetaDataVector BmmMeta(const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor mat2 = stack_tensor(stack, 1);
  auto self_sizes = self.sizes();
  auto mat2_sizes = mat2.sizes();
  auto self_end_iter = self_sizes.end();
  auto mat2_end_iter = mat2_sizes.end();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();

  if ((self.dim() == 4 && mat2.dim() == 4) ||
      (self.dim() == 5 && mat2.dim() == 5)) {
    meta.shape = self_sizes.vec();
    meta.shape.back() = mat2_sizes.back();
    return {meta};
  }

  TORCH_CHECK(self.dim() == 3, "BMM Input1 should be 3D, but got ", self.dim())
  TORCH_CHECK(mat2.dim() == 3, "BMM Input2 should be 3D, but got ", mat2.dim())
  // Inner Dimentions
  // self tensor  = {b, n, m} eg, [4, 3, 2]
  // other tensor = {b, m, p} eg, [4, 2, 5]
  // output = {b, n, p}  eg, [4, 3, 5]
  TORCH_CHECK(
      (*(self_end_iter - 1) == *(mat2_end_iter - 2) &&
       *(self_end_iter - 3) == *(mat2_end_iter - 3)),
      "Expected size for first two dimensions of batch2 tensor to be:",
      "[",
      *(self_end_iter - 3),
      ",",
      *(self_end_iter - 1),
      "] but got: [",
      *(mat2_end_iter - 3),
      ",",
      *(mat2_end_iter - 2),
      "]")

  meta.shape = {self_sizes[0], self_sizes[1], *(mat2_end_iter - 1)};
  return {meta};
}

} // namespace habana
