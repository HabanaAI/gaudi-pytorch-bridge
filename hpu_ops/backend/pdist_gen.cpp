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

#include "generated/backend/_pdist_backward.h"
#include "generated/backend/_pdist_forward.h"
#include "hpu_ops/op_backend.h"

namespace habana {
std::shared_ptr<void> FillPdistFwdParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Pdist::Params);
  params->p = stack.at(1).toScalar().toDouble();
  return params;
}

std::shared_ptr<void> FillPdistBwdParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Pdist::Params);
  params->p = stack.at(2).toScalar().toDouble();
  return params;
}

OutputMetaDataVector PdistFwdMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto shape = self.sizes().vec();

  HABANA_ASSERT(
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

OutputMetaDataVector PdistBwdMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 1);
  auto shape = self.sizes().vec();

  OutputMetaDataVector metas(1);
  metas[0].shape = shape;
  metas[0].dtype = self.scalar_type();

  return metas;
}

SharedMetaDataVector PdistBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& self = stack_tensor(stack, 1);

  auto dtype = grad.scalar_type();
  SharedMetaData meta{"pdist_bwd"};
  meta.inputs_data = {{grad.dim(), dtype}, {self.dim(), dtype}};
  meta.outputs_data = {{self.dim(), dtype}};

  return {meta};
}

void PdistBwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto op = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(op[0]);
}
} // namespace habana
