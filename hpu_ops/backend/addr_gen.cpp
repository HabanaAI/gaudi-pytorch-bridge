/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */
#include "generated/backend/addr.h"

namespace habana {

OutputMetaDataVector AddRMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto vec1 = stack_tensor(stack, 1);
  auto vec2 = stack_tensor(stack, 2);
  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 1 || self.dim() == 0,
      "addr: Expected self to be 0-D, 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  TORCH_CHECK(vec1.dim() == 1, "addr: Expected vec1 to be 1-D");
  TORCH_CHECK(vec2.dim() == 1, "addr: Expected vec2 to be 1-D");
  std::vector<int64_t> outshape{
      vec1.sizes()[0], vec2.sizes()[0]}; // (n, 1)@(1, m) -> (n, m)

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = outshape;
  return {meta};
}

void AddR::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto vec1 = stack_tensor(stack, 1);
  auto vec2 = stack_tensor(stack, 2);
  const float alpha_val = stack.at(4).toScalar().toFloat();
  const float beta_val = stack.at(3).toScalar().toFloat();
  auto meta = AddRMeta(stack)[0];
  auto vec1_reshaped = ReshapeHelper(
      graph,
      syn_in(1),
      {1, vec1.sizes()[0], 1},
      meta.dtype); // (n,) -> (1, n, 1)

  auto vec2_reshaped = ReshapeHelper(
      graph,
      syn_in(2),
      {1, 1, vec2.sizes()[0]},
      meta.dtype); // (m,) -> (1, 1, m)

  synGEMMParams vecmul_params{};

  std::vector<int64_t> vecmul_outshape = {
      1, vec1.sizes()[0], vec2.sizes()[0]}; // (1, n, 1)@(1, 1, m) -> (1, n, m)

  // Matrix Multiplication of vec1 and vec2
  auto vecmul = BuildOp(
      graph,
      "batch_gemm",
      {vec1_reshaped.get(), vec2_reshaped.get()},
      {{vecmul_outshape, meta.dtype}},
      &vecmul_params,
      sizeof(vecmul_params));

  auto alpha = ConstantHelper(graph, alpha_val, meta.dtype, vecmul_outshape);

  auto addr_unsqueezed = BuildOp(
      graph,
      get_guid_with_precision("mult", meta.dtype),
      {vecmul[0].get(), alpha.get()},
      {{vecmul_outshape, meta.dtype}});

  if (beta_val != 0.0) {
    auto self = stack_tensor(stack, 0);
    std::vector<int64_t> self_reshaped_outshape;
    if (self.dim() == 2) {
      self_reshaped_outshape = {1, self.sizes()[0], self.sizes()[1]};
    } else {
      self_reshaped_outshape = {1, 1, self.sizes()[0]};
    }
    std::vector<synapse_helpers::tensor> self_reshaped;
    self_reshaped.emplace_back(ReshapeHelper(
        graph,
        syn_in(0),
        self_reshaped_outshape,
        meta.dtype)); // (n,m) -> (1, n, m)

    if (beta_val != 1.0) {
      auto beta = ConstantHelper(graph, beta_val, meta.dtype, vecmul_outshape);

      self_reshaped = BuildOp(
          graph,
          get_guid_with_precision("mult", meta.dtype),
          {self_reshaped[0].get(), beta.get()},
          {{vecmul_outshape, meta.dtype}});
    }
    addr_unsqueezed = BuildOp(
        graph,
        get_guid_with_precision("add", meta.dtype),
        {self_reshaped[0].get(), addr_unsqueezed[0].get()},
        {{vecmul_outshape, meta.dtype}});
  }

  auto addr_out = BuildOp(
      graph,
      "squeeze",
      {addr_unsqueezed[0].get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(addr_out[0]);
}
} // namespace habana
