/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/addmm.h"
#define idxSelf 0
#define idxMat1 1
#define idxMat2 2
#define idxBeta 3
#define idxAlpha 4

namespace habana {

sizes_vec AddMMOutshape(const at::Stack& stack) {
  auto self = stack_tensor(stack, idxSelf);
  auto mat1 = stack_tensor(stack, idxMat1);
  auto mat2 = stack_tensor(stack, idxMat2);
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

void AddMM::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = AddMMOutshape(stack)[0];

  std::vector<synapse_helpers::tensor> addmm_out, beta_out, alpha_out;

  const float beta_val = stack.at(idxBeta).toScalar().toFloat();
  const float alpha_val = stack.at(idxAlpha).toScalar().toFloat();

  if (alpha_val == 0.0 && beta_val == 0.0) {
    addmm_out.emplace_back(
        ConstantHelper(graph, 0.0, ScalarType(), outshape, 0));
    syn_out(0) = std::move(addmm_out[0]);
  } else {
    if (beta_val != 0) {
      auto beta_tensor =
          ConstantHelper(graph, beta_val, ScalarType(), outshape);
      NodeAttr::NodeOutputAttr node_output_attr = {outshape, ScalarType()};
      if (alpha_val == 0.0)
        node_output_attr.final_result_index = 0;
      beta_out = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
          {syn_in(0), beta_tensor.get()},
          {node_output_attr});
    }

    if (alpha_val != 0.0) {
      synGEMMParams matmul_params{};
      alpha_out = BuildOp(
          graph,
          "gemm",
          {syn_in(1), syn_in(2)},
          {{outshape, ScalarType()}},
          &matmul_params,
          sizeof(matmul_params));

      if (alpha_val != 1.0) {
        auto alpha_tensor =
            ConstantHelper(graph, alpha_val, ScalarType(), outshape);
        NodeAttr::NodeOutputAttr node_output_attr = {outshape, ScalarType()};
        if (beta_val == 0.0)
          node_output_attr.final_result_index = 0;
        alpha_out = BuildOp(
            graph,
            MULT_GUID + habana_helpers::name_suffix_from_type(ScalarType()),
            {alpha_out[0].get(), alpha_tensor.get()},
            {node_output_attr});
      }
    }
    if (beta_val == 0.0)
      syn_out(0) = std::move(alpha_out[0]);
    else if (alpha_val == 0.0)
      syn_out(0) = std::move(beta_out[0]);
    else {
      addmm_out = BuildOp(
          graph,
          "add_" + habana_helpers::name_suffix_from_type(ScalarType()),
          {beta_out[0].get(), alpha_out[0].get()},
          {{outshape, ScalarType(), 0}});
      syn_out(0) = std::move(addmm_out[0]);
    }
  }
}
} // namespace habana
