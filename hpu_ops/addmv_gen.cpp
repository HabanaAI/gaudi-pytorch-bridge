/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/addmv.h"

#define idxSelf 0
#define idxMat1 1
#define idxMat2 2
#define idxBatch1 1
#define idxBatch2 2
#define idxBeta 3
#define idxAlpha 4

namespace habana {

sizes_vec AddMVOutshape(const at::Stack& stack) {
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

void AddMV::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = AddMVOutshape(stack)[0];

  const float beta_val = stack.at(idxBeta).toScalar().toFloat();
  const float alpha_val = stack.at(idxAlpha).toScalar().toFloat();

  auto beta_tensor = ConstantHelper(graph, beta_val, ScalarType(), 1);
  auto alpha_tensor = ConstantHelper(graph, alpha_val, ScalarType(), 1);

  auto addmv = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1), syn_in(2), beta_tensor.get(), alpha_tensor.get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(addmv[0]);
}

} // namespace habana
