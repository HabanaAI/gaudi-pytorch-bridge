/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/hpu_op.h"

namespace habana {

sizes_vec AddROutshape(const at::Stack& stack, bool) {
  auto self = stack_tensor(stack, 0);
  auto vec1 = stack_tensor(stack, 1);
  auto vec2 = stack_tensor(stack, 2);
  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 1,
      "addr: Expected self to be 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  TORCH_CHECK(vec1.dim() == 1, "addr: Expected vec1 to be 1-D");
  TORCH_CHECK(vec2.dim() == 1, "addr: Expected vec2 to be 1-D");
  std::vector<int64_t> outshape{
      vec1.sizes()[0], vec2.sizes()[0]}; // (n, 1)@(1, m) -> (n, m)
  return {outshape};
}

void AddR::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto vec1 = stack_tensor(stack, 1);
  auto vec2 = stack_tensor(stack, 2);
  const float alpha_val = stack.at(4).toScalar().toFloat();
  const float beta_val = stack.at(3).toScalar().toFloat();

  auto vec1_reshaped = BuildOp(
      graph,
      "reshape",
      {syn_in(1)},
      {{{1, vec1.sizes()[0], 1}, ScalarType()}}); // (n,) -> (1, n, 1)

  auto vec2_reshaped = BuildOp(
      graph,
      "reshape",
      {syn_in(2)},
      {{{1, 1, vec2.sizes()[0]}, ScalarType()}}); // (m,) -> (1, 1, m)

  synGEMMParams vecmul_params{};

  std::vector<int64_t> vecmul_outshape = {
      1, vec1.sizes()[0], vec2.sizes()[0]}; // (1, n, 1)@(1, 1, m) -> (1, n, m)

  // Matrix Multiplication of vec1 and vec2
  auto vecmul = BuildOp(
      graph,
      "batch_gemm",
      {vec1_reshaped[0].get(), vec2_reshaped[0].get()},
      {{vecmul_outshape, ScalarType()}},
      &vecmul_params,
      sizeof(vecmul_params));

  auto alpha = ConstantHelper(graph, alpha_val, ScalarType(), vecmul_outshape);

  auto addr_unsqueezed = BuildOp(
      graph,
      MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
      {vecmul[0].get(), alpha.get()},
      {{vecmul_outshape, ScalarType()}});

  if (beta_val != 0.0) {
    auto self = stack_tensor(stack, 0);
    std::vector<int64_t> self_reshaped_outshape;
    if (self.dim() == 2) {
      self_reshaped_outshape = {1, self.sizes()[0], self.sizes()[1]};
    } else {
      self_reshaped_outshape = {1, 1, self.sizes()[0]};
    }
    auto self_reshaped = BuildOp(
        graph,
        "reshape",
        {syn_in(0)},
        {{self_reshaped_outshape, ScalarType()}}); // (n,m) -> (1, n, m)

    if (beta_val != 1.0) {
      auto beta =
          ConstantHelper(graph, beta_val, ScalarType(), vecmul_outshape);

      self_reshaped = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
          {self_reshaped[0].get(), beta.get()},
          {{vecmul_outshape, ScalarType()}});
    }
    addr_unsqueezed = BuildOp(
        graph,
        "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {self_reshaped[0].get(), addr_unsqueezed[0].get()},
        {{vecmul_outshape, ScalarType()}});
  }
  auto outshape = AddROutshape(stack)[0];

  auto addr_out = BuildOp(
      graph,
      "squeeze",
      {addr_unsqueezed[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], 0}});

  syn_out(0) = std::move(addr_out[0]);
}
} // namespace habana
