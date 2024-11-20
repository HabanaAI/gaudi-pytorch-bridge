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
#include "generated/backend/addr.h"

namespace habana {

std::shared_ptr<void> FillAddrParams(const at::Stack& stack, size_t& size) {
  constexpr int BETA_INDEX = 3;
  constexpr int ALPHA_INDEX = 4;

  PARAMS_STUB(ns_AddrKernel::Params);

  params->beta = stack.at(BETA_INDEX).toScalar().toFloat();
  params->alpha = stack.at(ALPHA_INDEX).toScalar().toFloat();

  return params;
}

OutputMetaDataVector AddRMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto vec1 = stack_tensor(stack, 1);
  auto vec2 = stack_tensor(stack, 2);
  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 1 || self.dim() == 0,
      "addr: Expected self to be 0-D, 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  TORCH_CHECK(vec1.dim() == 1, "addr: Expected vec1 to be 1-D");
  TORCH_CHECK(vec2.dim() == 1, "addr: Expected vec2 to be 1-D");
  std::vector<int64_t> outshape{
      vec1.sizes()[0], vec2.sizes()[0]}; // (n, 1)@(1, m) -> (n, m)

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = outshape;
  return {meta};
}

} // namespace habana
