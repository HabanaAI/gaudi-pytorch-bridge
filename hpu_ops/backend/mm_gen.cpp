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

#include "generated/backend/mm.h"
namespace habana {
OutputMetaDataVector MmMeta(const at::Stack& stack) {
  TORCH_CHECK(
      (stack.at(0).isTensor() && stack.at(1).isTensor()),
      " Matmul Input type expected to be tensors");
  auto mat1 = stack.at(0).toTensor();
  auto mat2 = stack.at(1).toTensor();
  TORCH_CHECK(
      mat1.scalar_type() == mat2.scalar_type(),
      "expected m1 and m2 to have the same dtype, but got: ",
      mat1.scalar_type(),
      " != ",
      mat2.scalar_type());

  OutputMetaData meta;
  meta.shape = {mat1.size(0), mat2.size(1)};
  meta.dtype = mat1.scalar_type();
  return {meta};
}
} // namespace habana
