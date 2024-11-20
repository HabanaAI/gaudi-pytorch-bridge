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

#include "generated/backend/_masked_scale.h"

namespace habana {

OutputMetaDataVector MaskedScaleMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const torch::Tensor& self = stack_tensor(stack, 0);
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return {meta};
}

SharedMetaDataVector MaskedScaleSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto& mask = stack_tensor(stack, 1);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();

  SharedMetaData multSharedMeta{"mult_fwd"};
  multSharedMeta.inputs_data = {{rank, dtype}, {mask.dim(), dtype}};
  multSharedMeta.outputs_data.emplace_back(rank, dtype);

  if (rank > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(rank, dtype);
    return {multSharedMeta, constantSharedMeta};
  }
  return {multSharedMeta};
}

void MaskedScale::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto mask = stack.at(1).toTensor();
  auto scale = stack.at(2).toScalar().toDouble();
  scale = 1.0 / (1.0 - 1.0 / scale);
  const auto meta = MaskedScaleMeta(stack)[0];

  auto mult = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", meta.dtype),
      {syn_in(0), syn_in(1)},
      {{meta.shape, meta.dtype}});

  auto scale_tensor = ConstantHelper(graph, scale, meta.dtype, meta.shape);

  auto output = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", meta.dtype),
      {mult[0].get(), scale_tensor.get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(output[0]);
}
} // namespace habana
