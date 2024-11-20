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

#include "generated/backend/gather.h"

namespace habana {

OutputMetaDataVector GatherMeta(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto index = stack.at(2).toTensor();
  auto dim_ = stack.at(1).toInt();
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  std::vector<int64_t> shape = self.sizes().vec();
  // gather shape check
  auto self_dims = std::max<int64_t>(1, self.dim());
  auto index_dims = std::max<int64_t>(1, index.dim());
  TORCH_CHECK(
      self_dims == index_dims,
      "Index tensor must have the same number of dimensions as input tensor");
  for (int64_t i = 0; i < self_dims; ++i) {
    if (i != dim) {
      auto index_size = index.dim() == 0 ? 1 : index.size(i);
      auto self_size = self.dim() == 0 ? 1 : self.size(i);
      TORCH_CHECK(
          index_size <= self_size,
          "Size does not match at dimension ",
          i,
          " expected index ",
          index.sizes(),
          " to be smaller than self ",
          self.sizes(),
          " apart from dimension ",
          dim_);
    }
  }
  if (shape.size()) {
    // for gather op, output size is same as index
    if (self.dim() == index.dim()) {
      shape = index.sizes().vec();
    } else {
      // for index_select and other index ops
      shape[dim] = index.numel();
    }
  }
  OutputMetaData meta;
  meta.shape = shape;
  meta.dtype = self.scalar_type();
  return {meta};
}

SharedMetaDataVector GatherSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack_tensor(stack, 0);
  auto selfDtype = self.scalar_type();
  auto index = stack_tensor(stack, 2);
  auto rank = self.dim();
  SharedMetaData gatherElementsMeta{"gather_elements_fwd"};
  gatherElementsMeta.inputs_data = {
      {rank, selfDtype}, {rank, index.scalar_type()}};
  gatherElementsMeta.outputs_data = {{rank, selfDtype}};
  return {gatherElementsMeta};
}

std::shared_ptr<void> FillGatherParams(const at::Stack& stack, size_t& size) {
  auto self = stack.at(0).toTensor();
  int dim_ = stack.at(1).toInt();
  auto dim = get_dim_in_tpc_order(dim_, self.dim());
  at::Tensor indices = stack.at(2).toTensor();
  if (self.dim() != indices.dim()) {
    PARAMS_STUB(ns_GatherKernel::Params);
    params->axis = dim;
    return params;
  }
  PARAMS_STUB(ns_GatherElementsKernel::Params);
  params->axis = dim;
  return params;
}

void GatherElementsOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto index = stack.at(2).toTensor();

  synTensor index_val;
  std::unique_ptr<synapse_helpers::tensor> index_casted;
  if (common::IsInt64Supported() &&
      index.scalar_type() == c10::ScalarType::Long) {
    index_casted = std::make_unique<synapse_helpers::tensor>(BuildCast(
        this,
        graph,
        syn_in(1),
        index.sizes(),
        index.scalar_type(),
        torch::kInt));
    index_val = index_casted->get();
  } else {
    index_val = syn_in(1);
  }

  auto meta = GatherMeta(stack)[0];

  size_t params_size = 0;
  const auto& gather_params = FillGatherParams(stack, params_size);
  auto gatherOp = BuildOp(
      graph,
      get_guid_with_precision("gather_elements_fwd", ScalarType()),
      {syn_in(0), index_val},
      {{meta.shape, meta.dtype, 0}},
      gather_params.get(),
      params_size);
  syn_out(0) = std::move(gatherOp[0]);
}

} // namespace habana
