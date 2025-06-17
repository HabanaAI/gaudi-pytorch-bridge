/**
 * Copyright (c) 2024-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "generated/backend/linalg_qr.h"

namespace habana {

QRMode_t GetQrMode(std::string_view mode_str) {
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

FillParamsT FillQrParams(const at::Stack& stack) {
  PARAMS_STUB(ns_Qr::Params);
  auto modeString = stack.at(1).toStringView();
  params->mode = GetQrMode(modeString);
  return paramsT;
}
} // namespace habana
