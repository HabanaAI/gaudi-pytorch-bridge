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

#include "generated/backend/logical_and.h"
#include "generated/backend/logical_not.h"
#include "generated/backend/logical_or.h"
#include "generated/backend/logical_xor.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

OutputMetaDataVector LogicalMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = at::infer_size(
      stack.at(0).toTensor().sizes(), stack.at(1).toTensor().sizes());
  meta.dtype = at::kBool;

  return {meta};
}

OutputMetaDataVector LogicalNotMeta(const at::Stack& stack) {
  OutputMetaData meta;

  meta.shape = stack.at(0).toTensor().sizes().vec();
  meta.dtype = at::kBool;

  return {meta};
}

SharedMetaDataVector LogicalNotSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode) {
  auto self = stack.at(0).toTensor();
  auto rank = self.dim();
  auto dtype = self.scalar_type();
  SharedMetaDataVector metaVec = {};
  if ((self.scalar_type() == at::kFloat) ||
      (self.scalar_type() == at::kBFloat16) ||
      (self.scalar_type() == at::kInt) || (self.scalar_type() == at::kShort)) {
    metaVec = BoolCastSharedMeta({self}, executionMode);
    dtype = at::kBool;
  }

  SharedMetaData notMeta{"not"};
  notMeta.inputs_data = {{rank, dtype}};
  notMeta.outputs_data = {{rank, at::kBool}};
  metaVec.push_back(notMeta);
  return metaVec;
}

SharedMetaDataVector LogicalBinaryAndSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return LogicalBinarySharedMeta(stack, "and");
}

SharedMetaDataVector LogicalBinaryOrSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return LogicalBinarySharedMeta(stack, "or");
}

SharedMetaDataVector LogicalBinaryXorSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return LogicalBinarySharedMeta(stack, "xor");
}

void LogicalNotOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = LogicalNotMeta(stack).at(0);
  auto self = stack.at(0).toTensor();

  std::optional<synapse_helpers::tensor> castedInput{};
  if (self.scalar_type() == at::kFloat or self.scalar_type() == at::kBFloat16 or
      self.scalar_type() == at::kInt or self.scalar_type() == at::kShort) {
    castedInput =
        BuildBoolCast(this, graph, syn_in(0), self.sizes(), self.scalar_type());

    update_guid_dtype(guid_, at::kBool);
  }

  syn_out(0) = std::move(BuildOp(
      graph,
      guid_,
      {castedInput.has_value() ? castedInput.value().get() : syn_in(0)},
      {{meta.shape, meta.dtype, 0}})[0]);
}

} // namespace habana
