/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "hpu_ops/common/replication_pad.h"

namespace habana {
sizes_vec ComputePadOutputShape(const at::Stack& stack, PadType padType) {
  auto self = stack.at(0).toTensor();
  auto padding = stack.at(1).toIntVector();
  std::vector<int64_t> outputSize = self.sizes().vec();
  auto paddingSize = padding.size();
  auto selfRank = self.dim();
  TORCH_CHECK(
      paddingSize == 1 || paddingSize % 2 == 0,
      "Padding length must be divisible by 2");
  TORCH_CHECK(floor(paddingSize / 2) <= selfRank, "Padding length too large");
  TORCH_CHECK(
      (paddingSize == 1 || paddingSize == 2 || paddingSize == 4 ||
       paddingSize == 6) &&
          (selfRank >= 2 && selfRank <= 5),
      "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now");

  if (paddingSize == 1)
    padding.resize((padType + 1) * 2);
  for (auto i = 0; i <= padType; i++)
    outputSize.rbegin()[i] += padding.at(i * 2) + padding.at(i * 2 + 1);

  return {outputSize};
}

std::shared_ptr<void> FillPadFwdBwdParams(
    const at::Stack& stack,
    PadType padType,
    size_t& size,
    bool backward) {
  PARAMS_STUB(ns_PadKernelEx::Params);
  size_t offset = backward ? 1 : 0;
  auto self = stack.at(offset).toTensor();
  auto padding = stack.at(1 + offset).toIntVector();
  auto selfRank = self.dim();
  params->mode = PadMode_t::PAD_MODE_EDGE;

  if (padding.size() == 1)
    padding.resize((padType + 1) * 2);
  for (auto i = 0; i <= padType; i += 1) {
    params->pads[i] = padding.at(i * 2);
    params->pads[i + selfRank] = padding.at(i * 2 + 1);
  }

  return params;
}

} // namespace habana