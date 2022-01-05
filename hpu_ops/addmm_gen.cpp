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

sizes_vec AddMMOutshape(const at::Stack& stack, bool) {
  auto self = stack_tensor(stack, 0);
  auto mat1 = stack_tensor(stack, 1);
  auto mat2 = stack_tensor(stack, 2);
  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 1,
      "addmm: Expected self to be 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  TORCH_CHECK(
      mat1.dim() == 2,
      "addmm: Expected mat1 to be 2-D, but got ",
      mat1.dim(),
      "-D");
  TORCH_CHECK(
      mat2.dim() == 2,
      "addmm: Expected mat2 to be 2-D, but got ",
      mat2.dim(),
      "-D");
  std::vector<int64_t> outshape{
      mat1.sizes()[0], mat2.sizes()[1]}; // (n, m)@(m, p) -> (n, p)
  return {outshape};
}

void AddMM::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto outshape = AddMMOutshape(stack)[0];

  auto mat1 = stack_tensor(stack, 1);
  auto mat2 = stack_tensor(stack, 2);
  const float alpha_val = stack.at(4).toScalar().toFloat();
  const float beta_val = stack.at(3).toScalar().toFloat();

  synGEMMParams matmul_params{};

  // Matrix Multiplication of mat1 and mat2
  auto addmm = BuildOp(
      graph,
      "gemm",
      {syn_in(1), syn_in(2)},
      {{outshape, ScalarType()}},
      &matmul_params,
      sizeof(matmul_params));

  std::vector<int64_t> addmm_reshaped_outshape{
      1, mat1.sizes()[0], mat2.sizes()[1]};
  addmm = BuildOp(
      graph,
      "reshape",
      {addmm[0].get()},
      {{addmm_reshaped_outshape, ScalarType()}}); // (n, p) -> (1, n, p)

  if (alpha_val != 1.0) {
    auto alpha =
        ConstantHelper(graph, alpha_val, ScalarType(), addmm_reshaped_outshape);

    addmm = BuildOp(
        graph,
        MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
        {addmm[0].get(), alpha.get()},
        {{addmm_reshaped_outshape, ScalarType()}});
  }

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
        {{self_reshaped_outshape, ScalarType()}}); // (n, p) -> (1, n, p)

    if (beta_val != 1.0) {
      auto beta = ConstantHelper(
          graph, beta_val, ScalarType(), addmm_reshaped_outshape);
      self_reshaped = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
          {self_reshaped[0].get(), beta.get()},
          {{addmm_reshaped_outshape, ScalarType()}});
    }
    addmm = BuildOp(
        graph,
        "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {self_reshaped[0].get(), addmm[0].get()},
        {{addmm_reshaped_outshape, ScalarType()}});
  }

  // Output
  auto addmm_out = BuildOp(
      graph,
      "squeeze",
      {addmm[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  syn_out(0) = std::move(addmm_out[0]);
}
} // namespace habana
