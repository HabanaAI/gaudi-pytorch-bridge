/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

std::shared_ptr<void> FillAddrParams(const at::Stack& stack, size_t& size) {
  constexpr int BETA_INDEX = 3;
  constexpr int ALPHA_INDEX = 4;

  PARAMS_STUB(ns_AddrKernel::Params);

  params->beta = stack.at(BETA_INDEX).toScalar().toFloat();
  params->alpha = stack.at(ALPHA_INDEX).toScalar().toFloat();

  return params;
}

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

} // namespace habana
