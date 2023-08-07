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
 *******************************************************************************
 */

#include "generated/backend/constant_pad_nd.h"

namespace habana {

std::vector<int64_t> pad_output_shape(
    const at::Tensor& self,
    c10::IntArrayRef pad) {
  auto ndim = self.dim();
  auto lpad = pad.size() / 2;

  TORCH_CHECK(
      pad.size() % 2 == 0,
      "Length of pad must be even but instead it equals ",
      pad.size());

  TORCH_CHECK(
      ndim >= (int64_t)lpad,
      "Length of pad should be no more than twice the number of "
      "dimensions of the input. Pad length is ",
      pad.size(),
      "while the input has ",
      ndim,
      "dimensions.");

  auto shape = self.sizes().vec();

  for (unsigned int i = 0; i < lpad; i++) {
    auto pad_start = pad[2 * i];
    auto pad_end = pad[2 * i + 1];
    shape[ndim - i - 1] += (pad_start + pad_end);
    TORCH_CHECK(
        shape[ndim - i - 1] > 0,
        "The input size ",
        self.sizes()[i],
        ", plus negative padding ",
        pad_start,
        " and ",
        pad_end,
        " resulted in a invalid output size, "
        "Check dimension ",
        i,
        " of your input.");
  }
  return shape;
}

OutputMetaDataVector ConstantPadMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  auto pad = stack.at(1).toIntVector();
  OutputMetaData meta;
  meta.shape = pad_output_shape(self, pad);
  meta.mem_format = self.suggest_memory_format();
  meta.dtype = self.scalar_type();
  return {meta};
}

std::shared_ptr<void> FillConstantPadParams(
    const at::Stack& stack,
    size_t& size) {
  auto self = stack.at(0).toTensor();
  auto pad = stack.at(1).toIntVector();

  auto ndim = self.dim();
  auto lpad = pad.size() / 2;

  PARAMS_STUB(ns_PadKernelEx::Params);

  params->mode = PadMode_t::PAD_MODE_CONSTANT;
  if (c10::isIntegralType(self.scalar_type(), false)) {
    params->value.i = stack.at(2).toScalar().to<decltype(params->value.i)>();
  } else {
    params->value.f = stack.at(2).toScalar().to<float>();
  }
  memset(params->pads, 0, sizeof(params->pads));
  for (unsigned int i = 0; i < lpad; i++) {
    params->pads[i] = pad[2 * i];
    params->pads[i + ndim] = pad[2 * i + 1];
  }
  return params;
}

} // namespace habana