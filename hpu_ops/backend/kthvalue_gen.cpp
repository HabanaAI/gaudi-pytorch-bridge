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

#include "generated/backend/kthvalue.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

std::vector<int64_t> KthvalueOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  int axis = stack.at(2).toInt();
  bool keep_dims = stack.at(3).toBool();

  return ReductionOutputShape(self, axis, keep_dims)[0];
}

std::shared_ptr<void> FillKthvalueParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Kthvalue::Params);
  params->k_value = stack.at(1).toInt();
  params->axis =
      get_dim_in_tpc_order(stack.at(2).toInt(), stack_tensor(stack, 0).dim());
  params->keep_dims = stack.at(3).toBool();

  return params;
}

OutputMetaDataVector KthvalueMeta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  auto output_shape = KthvalueOutputShape(stack);

  OutputMetaData values_meta, indices_meta;
  values_meta.dtype = input.scalar_type();
  values_meta.shape = output_shape;
  indices_meta.dtype = c10::ScalarType::Long;
  indices_meta.shape = output_shape;
  return {values_meta, indices_meta};
}

} // namespace habana
