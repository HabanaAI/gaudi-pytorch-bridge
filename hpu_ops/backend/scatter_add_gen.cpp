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
#include "generated/backend/scatter_add.h"
namespace habana {

const unsigned SELF_INDEX = 0;
const unsigned DIM_INDEX = 1;
const unsigned IND_INDEX = 2;
const unsigned SRC_INDEX = 3;

std::shared_ptr<void> ScatterAddParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_ScatterKernel::ParamsV2);
  const auto dim = stack.at(DIM_INDEX).toInt();

  const auto rank = stack.at(SELF_INDEX).toTensor().dim();
  params->dim = dim;
  params->axis = get_dim_in_tpc_order(dim, rank);

  return params;
}

OutputMetaDataVector ScatterAddMeta(const at::Stack& stack) {
  const auto selfTensor = stack.at(SELF_INDEX).toTensor();
  const auto dim = stack.at(DIM_INDEX).toInt();
  const auto indexTensor = stack.at(IND_INDEX).toTensor();
  const auto srcTensor = stack.at(SRC_INDEX).toTensor();

  auto index_dims = indexTensor.dim();
  int dim_ = at::maybe_wrap_dim(dim, selfTensor.dim());

  // https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html
  for (int64_t d = 0; d < index_dims; ++d) {
    HABANA_ASSERT(
        srcTensor.size(d) >= indexTensor.size(d),
        "index.size(d) > src.size(d) at d = ",
        d);
    if (d != dim_) {
      HABANA_ASSERT(
          selfTensor.size(d) >= indexTensor.size(d),
          "index.size(d) > self.size(d) at d = ",
          d);
    }
  }

  OutputMetaData meta;
  meta.dtype = selfTensor.scalar_type();
  meta.shape = selfTensor.sizes().vec();

  return {meta};
}
} // namespace habana