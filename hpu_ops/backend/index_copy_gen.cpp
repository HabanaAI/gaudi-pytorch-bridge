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
#include "generated/backend/index_copy.h"

namespace habana {

OutputMetaDataVector IndexCopyMeta(const at::Stack& stack) {
  const auto& input_tensor = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  const auto& copy_tensor = stack.at(3).toTensor();
  auto inputTensorShape = input_tensor.sizes().vec();
  auto copyTensorShape = copy_tensor.sizes().vec();

  dim = at::maybe_wrap_dim(dim, input_tensor.dim(), /*wrap_scalar=*/true);
  inputTensorShape.erase(inputTensorShape.begin() + dim);
  copyTensorShape.erase(copyTensorShape.begin() + dim);
  TORCH_CHECK(
      inputTensorShape == copyTensorShape,
      " Source/destination tensor must have same slice shapes except at dimension ",
      dim,
      " Destination slice shape: ",
      input_tensor.sizes().vec(),
      " and source slice shape: ",
      copy_tensor.sizes().vec());
  OutputMetaData meta;
  meta.dtype = input_tensor.scalar_type();
  meta.shape = input_tensor.sizes().vec();
  return {meta};
}

std::shared_ptr<void> FillIndexCopyParams(
    const at::Stack& stack,
    size_t& size) {
  const auto dim = stack[1].toInt();
  PARAMS_STUB(ns_IndexCopy::Params);
  params->axis = dim;
  return params;
}

} // namespace habana
