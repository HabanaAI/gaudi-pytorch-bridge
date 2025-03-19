/*******************************************************************************
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

#include "generated/backend/linalg_qr.h"

namespace habana {

QRMode_t GetQrMode(c10::string_view mode_str) {
  if (mode_str == "complete") {
    return QRMode_t::COMPLETE;
  } else if (mode_str == "r") {
    return QRMode_t::R;
  } else if (mode_str == "reduced") {
    return QRMode_t::REDUCED;
  } else {
    HABANA_ASSERT(false, "Invalid QR mode: ", mode_str);
  }
}

OutputMetaDataVector QrMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> selfShape = self.sizes().vec();
  const auto modeString = stack.at(1).toStringView();

  auto selfDims = selfShape.size();
  HABANA_ASSERT(selfDims >= 2, "Input tensor must be at least 2D");
  auto m = selfShape[selfDims - 2];
  auto n = selfShape[selfDims - 1];
  auto k = std::min(m, n);

  std::vector<int64_t> qShape(selfShape);
  std::vector<int64_t> rShape(selfShape);

  auto mode = GetQrMode(modeString);
  switch (mode) {
    case QRMode_t::COMPLETE:
      qShape[selfDims - 1] = m;
      break;
    case QRMode_t::REDUCED:
      qShape[selfDims - 1] = k;
      rShape[selfDims - 2] = k;
      break;
    case QRMode_t::R:
      qShape = std::vector<int64_t>{0};
      rShape[selfDims - 2] = k;
      break;
    default:
      HABANA_ASSERT(false, "Invalid QR mode: ", modeString);
  }

  return OutputMetaDataVector{
      OutputMetaData(self.scalar_type(), qShape),
      OutputMetaData(self.scalar_type(), rShape)};
}

std::shared_ptr<void> FillQrParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Qr::Params);
  auto modeString = stack.at(1).toStringView();
  params->mode = GetQrMode(modeString);
  return params;
}
} // namespace habana
