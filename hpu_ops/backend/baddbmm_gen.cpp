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

#include "generated/backend/baddbmm.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

OutputMetaDataVector BaddbmmMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto batch1 = stack_tensor(stack, 1);
  auto batch2 = stack_tensor(stack, 2);
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();
  HABANA_ASSERT(batch1.dim() == 3, "batch1 must be a 3D tensor");
  HABANA_ASSERT(batch2.dim() == 3, "batch2 must be a 3D tensor");
  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = {bs, res_rows, res_cols};
  meta.dtype = self.scalar_type();

  HABANA_ASSERT(
      batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
      "Expected size for first two dimensions of batch2 tensor to be: [",
      bs,
      ", ",
      contraction_size,
      "] but got: [",
      batch2_sizes[0],
      ", ",
      batch2_sizes[1],
      "].");

  return metaVec;
}

SharedMetaDataVector BAddBMMSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return MatrixMulWithAddSharedMeta(stack, "baddbmm", false);
}

void Baddbmm::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto meta = OutputMeta(stack);

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

  const bool shouldUseParams = beta_val == 0.0 || beta_val == 1.0 ||
      alpha_val == 1.0 || alpha_val == 0.0;

  // Kernel precision type is based on the input1 of the addmm op,
  // because we want to support configuration: inputs(fp8), output(bf16/fp32).
  // Formula: out = beta * input0 + alpha * (input1 @ input2)
  // GEMM returns higher precision dtype, so input0 has to be (bf16/fp32).
  update_guid_dtype(guid_, stack_tensor(stack, 1).scalar_type());

  if (shouldUseParams) {
    ns_AddmmKernel::Params params{};
    params.alpha = alpha_val;
    params.beta = beta_val;

    auto baddbmmv = BuildOp(
        graph,
        GetGuid(),
        {syn_in(0), syn_in(1), syn_in(2)},
        {{meta[0].shape, meta[0].dtype, 0}},
        &params,
        sizeof(params));
    syn_out(0) = std::move(baddbmmv[0]);
  } else {
    auto alpha_tensor = ConstantHelper(graph, alpha_val, ScalarType(), 1);
    auto beta_tensor = ConstantHelper(graph, beta_val, ScalarType(), 1);
    auto baddbmm = BuildOp(
        graph,
        GetGuid(),
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         beta_tensor.get(),
         alpha_tensor.get()},
        {{meta[0].shape, meta[0].dtype, 0}});

    syn_out(0) = std::move(baddbmm[0]);
  }
}
} // namespace habana
