/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/addr.h"

namespace habana {

sizes_vec AddROutshape(const at::Stack& stack) {
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
  return {outshape};
}

std::shared_ptr<void> FillAddrParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_AddrKernel::Params);
  const float beta = stack.at(3).toScalar().toFloat();
  const float alpha = stack.at(4).toScalar().toFloat();
  params->beta = beta;
  params->alpha = alpha;
  return params;
}
} // namespace habana
