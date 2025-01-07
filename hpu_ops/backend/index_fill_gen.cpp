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
#include "generated/backend/index_fill.h"

namespace habana {

std::shared_ptr<void> FillIndexFillParams(
    const at::Stack& stack,
    size_t& size) {
  const auto dim =
      at::maybe_wrap_dim(stack.at(1).toInt(), stack.at(0).toTensor().dim());
  PARAMS_STUB(ns_IndexCopy::Params);
  params->axis = dim;
  return params;
}

OutputMetaDataVector IndexFillMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();

  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  return {meta};
}

SharedMetaDataVector IndexFillSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto selfRank = self.dim();
  const auto computeDtype = self.scalar_type();
  const auto& index = stack_tensor(stack, 2);
  const auto indexRank = index.dim();
  const auto indexDtype = index.scalar_type();
  const auto& value = stack.at(3);

  SharedMetaDataVector metaVec;
  if (value.isTensor()) {
    auto valueTensor = value.toTensor();
    auto valueDtype = valueTensor.scalar_type();
    SharedMetaData broadcastSharedMeta{"broadcast"};
    broadcastSharedMeta.inputs_data.emplace_back(1, valueDtype);
    broadcastSharedMeta.outputs_data.emplace_back(selfRank, valueDtype);
    metaVec.push_back(broadcastSharedMeta);
  } else if (selfRank > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(selfRank, computeDtype);
    metaVec.push_back(constantSharedMeta);
  }

  SharedMetaData indexFillSharedMeta{"index_copy_fwd"};
  indexFillSharedMeta.inputs_data = {
      {selfRank, computeDtype},
      {indexRank, indexDtype},
      {selfRank, computeDtype}};
  indexFillSharedMeta.outputs_data.emplace_back(selfRank, computeDtype);
  metaVec.push_back(indexFillSharedMeta);
  return metaVec;
}

void IndexFill::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& dim = stack.at(1).toInt();
  const auto& indexes = stack.at(2).toTensor();
  bool is_value_scalar = stack.at(3).isScalar();

  const auto meta = IndexFillMeta(stack)[0];

  auto valueTensorShape = input.sizes().vec();
  if (!valueTensorShape.empty()) {
    valueTensorShape[dim] = indexes.numel();
  } else {
    HABANA_ASSERT(
        indexes.numel() == 1,
        "For input 0-D tensor, number of elements in indices tensor should be 1.");
  }

  synapse_helpers::tensor valueTensor = is_value_scalar
      ? OpBackend::BuildConstant(
            this,
            graph,
            stack.at(3).toScalar().toFloat(),
            meta.dtype,
            valueTensorShape)
      : BroadcastHelper(
            graph,
            syn_in(2),
            valueTensorShape,
            stack.at(3).toTensor().scalar_type());

  size_t size = 0;
  const auto params = FillIndexFillParams(stack, size);

  auto indexCopyResult = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1), valueTensor.get()},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(indexCopyResult[0]);
}

} // namespace habana
