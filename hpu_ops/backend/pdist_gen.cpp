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

#include "generated/backend/_pdist_forward.h"
#include "hpu_ops/op_backend.h"

namespace habana {
std::shared_ptr<void> FillPdistFwdParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Pdist::Params);
  params->p = stack.at(1).toScalar().toDouble();
  return params;
}

OutputMetaDataVector PdistFwdMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shape = self.sizes().vec();

  TORCH_CHECK(
      shape.size() == 2,
      "pdist only supports 2D tensors, got: ",
      shape.size(),
      "D");

  auto d = shape[0];
  OutputMetaDataVector metas(1);
  metas[0].shape = {(d >= 2) ? d * (d - 1) / 2 : 0};
  metas[0].dtype = self.scalar_type();

  return metas;
}

} // namespace habana
