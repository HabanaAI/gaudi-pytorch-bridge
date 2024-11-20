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
#include "generated/backend/median.h"
#include "hpu_ops/topk_util.h"

namespace habana {
constexpr size_t index_of_self = 0;
constexpr size_t index_of_reduction_axis = 1;
constexpr size_t index_of_keepdim = 2;
constexpr int descending_order = 0;

std::shared_ptr<void> FillMediandimParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MediandimKernel::Params);

  params->reduction_dim = stack[index_of_reduction_axis].toInt();
  params->keep_dim = stack[index_of_keepdim].toBool();;

  return params;
}

OutputMetaDataVector MedianOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {};
  meta.dtype = stack_tensor(stack, 0).scalar_type();
  return {meta};
}

SharedMetaDataVector MedianSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);

  SharedMetaData medianSharedMeta{"median"};
  medianSharedMeta.inputs_data.emplace_back(self.dim(), self.scalar_type());
  medianSharedMeta.outputs_data.emplace_back(1, self.scalar_type());

  return {medianSharedMeta};
}

sizes_vec MediandimOutputShape(const at::Stack& stack) {
  auto self = stack.at(index_of_self).toTensor();
  auto self_size = self.sizes().vec();
  int64_t reduction_axis = c10::maybe_wrap_dim(
      stack[index_of_reduction_axis].toInt(),
      self.dim(),
      /*wrap_scalar=*/true);

  bool keepdim = stack[index_of_keepdim].toBool();
  std::vector<int64_t> outshape = {self_size};
  if (outshape.size() == 0) {
    return {outshape, outshape};
  }

  if (keepdim)
    outshape[reduction_axis] = 1;
  else {
    std::vector<int64_t>::iterator itr = outshape.begin() + reduction_axis;
    outshape.erase(itr);
  }
  return {outshape, outshape};
}

OutputMetaDataVector MedianDimOutputMeta(const at::Stack& stack) {
  auto medianDimShapes = MediandimOutputShape(stack);
  auto self = stack_tensor(stack, index_of_self);

  OutputMetaData valuesMeta, indicesMeta;

  valuesMeta.shape = medianDimShapes[0];
  valuesMeta.dtype = self.scalar_type();

  indicesMeta.shape = medianDimShapes[1];
  indicesMeta.dtype =
      common::IsInt64Supported() ? c10::ScalarType::Long : c10::ScalarType::Int;

  return {valuesMeta, indicesMeta};
}

SharedMetaDataVector MedianDimSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  bool keepDim = stack[index_of_keepdim].toBool();
  auto outputRank = self.dim();
  if (!keepDim)
    outputRank = outputRank <= 1 ? 1 : (outputRank - 1);

  SharedMetaData medianDimSharedMeta{"mediandim"};
  medianDimSharedMeta.inputs_data.emplace_back(self.dim(), self.scalar_type());
  medianDimSharedMeta.outputs_data.emplace_back(outputRank, self.scalar_type());
  medianDimSharedMeta.outputs_data.emplace_back(
      outputRank, c10::ScalarType::Int);

  return {medianDimSharedMeta};
}

void Mediandim::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack_tensor(stack, index_of_self);
  auto self_size = self.sizes().vec();

  size_t size = 0;
  auto params = FillMediandimParams(stack, size);
  auto meta = MedianDimOutputMeta(stack);

  auto result = BuildOp(
      graph,
      GetGuid(),
      {syn_in(0)},
      {{meta[0].shape, meta[0].dtype, 0}, {meta[1].shape, meta[1].dtype, 1}},
      params.get(),
      size);

  syn_out(0) = std::move(result[0]);
  syn_out(1) = std::move(result[1]);

  return;
}

} // namespace habana
