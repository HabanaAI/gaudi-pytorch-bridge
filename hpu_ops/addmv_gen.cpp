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

sizes_vec AddMVOutshape(const at::Stack& stack, bool) {
  auto self = stack_tensor(stack, 0);
  auto mat = stack_tensor(stack, 1);
  auto vec = stack_tensor(stack, 2);
  TORCH_CHECK(
      self.dim() == 1,
      "addmv: Expected self to be 1-D, but got ",
      self.dim(),
      "-D");
  TORCH_CHECK(
      mat.dim() == 2,
      "addmv: Expected mat to be 2-D, but got ",
      mat.dim(),
      "-D");
  TORCH_CHECK(
      vec.dim() == 1,
      "addmv: Expected vec to be 1-D, but got ",
      vec.dim(),
      "-D");
  std::vector<int64_t> outshape{mat.sizes()[0]}; // (n, m)@(m, 1) -> (n, 1)
  return {outshape};
}

void AddMV::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto mat = stack_tensor(stack, 1);
  auto vec = stack_tensor(stack, 2);
  const float alpha_val = stack.at(4).toScalar().toFloat();
  const float beta_val = stack.at(3).toScalar().toFloat();

  std::vector<int64_t> matvecmul_outshape = {
      1, mat.sizes()[0], 1}; // (1, n, m)@(1, m, 1) -> (1, n, 1)

  std::vector<synapse_helpers::tensor> addmv;
  if (alpha_val == 0.0) {
    addmv = BuildOp(graph, "memset", {}, {{matvecmul_outshape, ScalarType()}});
  } else {
    auto mat_reshaped = BuildOp(
        graph,
        "reshape",
        {syn_in(1)},
        {{{1, mat.sizes()[0], mat.sizes()[1]},
          ScalarType()}}); // (n, m) -> (1, n, m)

    auto vec_reshaped = BuildOp(
        graph,
        "reshape",
        {syn_in(2)},
        {{{1, vec.sizes()[0], 1}, ScalarType()}}); // (m,) -> (1, m, 1)

    synGEMMParams matvecmul_params{};

    std::vector<int64_t> matvecmul_outshape = {
        1, mat.sizes()[0], 1}; // (1, n, m)@(1, m, 1) -> (1, n, 1)

    // Matrix Multiplication of mat and vec
    addmv = BuildOp(
        graph,
        "batch_gemm",
        {mat_reshaped[0].get(), vec_reshaped[0].get()},
        {{matvecmul_outshape, ScalarType()}},
        &matvecmul_params,
        sizeof(matvecmul_params));
  }

  if (alpha_val != 1.0) {
    auto alpha =
        ConstantHelper(graph, alpha_val, ScalarType(), matvecmul_outshape);

    addmv = BuildOp(
        graph,
        "mult_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {addmv[0].get(), alpha.get()},
        {{matvecmul_outshape, ScalarType()}});
  }

  if (beta_val != 0.0) {
    auto self = stack_tensor(stack, 0);
    std::vector<int64_t> self_reshaped_outshape{1, self.sizes()[0], 1};
    auto self_reshaped = BuildOp(
        graph,
        "reshape",
        {syn_in(0)},
        {{self_reshaped_outshape, ScalarType()}}); // (n,) -> (1, n, 1)

    if (beta_val != 1.0) {
      auto beta =
          ConstantHelper(graph, beta_val, ScalarType(), matvecmul_outshape);

      self_reshaped = BuildOp(
          graph,
          "mult_" + habana_helpers::name_suffix_from_type(ScalarType()),
          {self_reshaped[0].get(), beta.get()},
          {{matvecmul_outshape, ScalarType()}});
    }
    addmv = BuildOp(
        graph,
        "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {self_reshaped[0].get(), addmv[0].get()},
        {{matvecmul_outshape, ScalarType()}});
  }
  auto outshape = AddMVOutshape(stack)[0];

  // Output
  auto addmv_out = BuildOp(
      graph,
      "squeeze",
      {addmv[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}});

  syn_out(0) = std::move(addmv_out[0]);
}
} // namespace habana
