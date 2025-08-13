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
#include "generated/backend/addmv.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

OutputMetaDataVector AddMVMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto mat = stack_tensor(stack, 1);
  auto vec = stack_tensor(stack, 2);
  HABANA_ASSERT(
      (mat.dim() == 2 && vec.dim() == 1 && self.dim() <= 1),
      "vector + matrix @ vector expected, got ",
      self.dim(),
      ", ",
      mat.dim(),
      ", ",
      vec.dim());

  HABANA_ASSERT(
      mat.size(1) == vec.size(0) &&
          (mat.size(0) == self.numel() || self.numel() == 1),
      "size mismatch, got ",
      self.size(0),
      ", ",
      mat.size(0),
      "x",
      mat.size(1),
      ",",
      vec.size(0));

  OutputMetaData meta;
  meta.dtype = mat.scalar_type();
  meta.shape = {mat.sizes()[0]}; // (n, m)@(m, 1) -> (n, 1)

  return {meta};
}

SharedMetaDataVector AddMVSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return MatrixMulWithAddSharedMeta(stack, "addmv", false);
}

void AddMV::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = AddMVMeta(stack)[0];
  update_guid_dtype(guid_, meta.dtype);
  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

  const bool shouldUseParams =
      beta_val == 0.0 || beta_val == 1.0 || alpha_val == 1.0;
  if (shouldUseParams) {
    ns_AddmvKernel::Params params{};
    params.alpha = alpha_val;
    params.beta = beta_val;

    auto addmv = BuildOp(
        graph,
        GetGuid(),
        {syn_in(0), syn_in(1), syn_in(2)},
        {{meta.shape, meta.dtype, 0}},
        &params,
        sizeof(params));
    syn_out(0) = std::move(addmv[0]);
  } else {
    auto alpha_tensor = ConstantHelper(graph, alpha_val, meta.dtype, 1);
    auto beta_tensor = ConstantHelper(graph, beta_val, meta.dtype, 1);
    auto addmm = BuildOp(
        graph,
        GetGuid(),
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         beta_tensor.get(),
         alpha_tensor.get()},
        {{meta.shape, meta.dtype, 0}});

    syn_out(0) = std::move(addmm[0]);
  }
}

} // namespace habana
