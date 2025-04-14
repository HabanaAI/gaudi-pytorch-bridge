/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "generated/backend/_cdist_backward.h"
#include "generated/backend/_cdist_forward.h"
#include "hpu_ops/common/batched_matmul_output_shape.h"
#include "hpu_ops/op_backend.h"

namespace habana {
std::shared_ptr<void> FillCdistFwdParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Cdist::Params);
  params->p = stack.at(2).toScalar().toDouble();
  c10::IValue cmVal = stack.at(3);
  params->compute_mode =
      static_cast<CdistComputeMode_t>(cmVal.isInt() ? cmVal.toInt() : 0);
  return params;
}

OutputMetaDataVector CdistFwdMeta(const at::Stack& stack) {
  OutputMetaDataVector metas(1);

  std::array<c10::IntArrayRef, 2> shapes;
  for (size_t i = 0; i < shapes.size(); ++i) {
    auto input = stack_tensor(stack, i);
    if (i == 0) {
      metas[0].dtype = input.scalar_type();
    }

    auto& shape = shapes[i];
    shape = input.sizes();

    HABANA_ASSERT(
        shape.size() >= 2,
        "Cdist only supports 2D tensors or above, got: ",
        shape.size(),
        "D");
  }

  metas[0].shape = getBatchMatmulOutShape(shapes[0], shapes[1], false, true);
  return metas;
}

std::shared_ptr<void> FillCdistBwdParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Cdist::Params);
  params->p = stack.at(3).toScalar().toDouble();
  return params;
}

OutputMetaDataVector CdistBwdMeta(const at::Stack& stack) {
  auto x1 = stack_tensor(stack, 1);
  auto x2 = stack_tensor(stack, 2);

  OutputMetaDataVector metas(1);
  metas[0].shape = x1.sizes().vec();
  metas[0].dtype = x1.scalar_type();

  TORCH_CHECK(
      x1.dim() == 2 && x2.dim() == 2,
      "Cdist for backward only supports 2D x1 and x2 tensors but got: ",
      x1.dim(),
      "D and ",
      x2.dim(),
      "D");

  auto grad_input_shape = stack_tensor(stack, 0).sizes();
  auto expected_grad_input_shape =
      getBatchMatmulOutShape(x1.sizes(), x2.sizes(), false, true);
  TORCH_CHECK(
      grad_input_shape == expected_grad_input_shape,
      "Cdist backward: expected grad_input shape ",
      expected_grad_input_shape,
      " but got ",
      grad_input_shape);

  return metas;
}

SharedMetaDataVector CdistBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& x1 = stack_tensor(stack, 1);
  const auto& x2 = stack_tensor(stack, 2);

  auto dtype = grad.scalar_type();
  SharedMetaData meta{"cdist_bwd"};
  meta.inputs_data = {
      {grad.dim(), dtype}, {x1.dim(), dtype}, {x2.dim(), dtype}};
  meta.outputs_data = {{x1.dim(), dtype}, {x2.dim(), dtype}};

  return {meta};
}

void CdistBwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = OutputMeta(stack);
  auto x2 = stack_tensor(stack, 2);
  size_t size = 0;
  auto params = FillParams(stack, size);
  auto op = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1), syn_in(2)},
      {{meta[0].shape, meta[0].dtype, 0}, {x2.sizes().vec(), x2.scalar_type()}},
      params.get(),
      size);
  syn_out(0) = std::move(op.at(0));
}
} // namespace habana
