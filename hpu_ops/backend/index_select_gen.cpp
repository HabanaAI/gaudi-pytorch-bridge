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
#include "generated/backend/index_select.h"

constexpr int64_t index_of_self = 0;
constexpr int64_t index_of_dim = 1;
constexpr int64_t index_of_index_position = 2;

namespace habana {

std::shared_ptr<void> FillIndexSelectParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_GatherKernel::Params);

  auto self = stack.at(index_of_self).toTensor();
  auto dim = stack.at(index_of_dim).toInt();
  params->axis = get_dim_in_tpc_order(dim, self.dim());
  return params;
}

OutputMetaDataVector IndexSelectMeta(const at::Stack& stack) {
  auto self = stack.at(index_of_self).toTensor();
  auto dim_ = stack.at(index_of_dim).toInt();
  auto index = stack.at(index_of_index_position).toTensor();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto shape = self.sizes().vec();

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  if (shape.size()) {
    if (self.dim() == index.dim()) {
      meta.shape = index.sizes().vec();
    } else {
      shape.erase(shape.begin() + dim);
      shape.insert(shape.begin() + dim, index.numel());
      meta.shape = shape;
    }
  } else {
    meta.shape = shape;
  }
  return {meta};
}
} // namespace habana
