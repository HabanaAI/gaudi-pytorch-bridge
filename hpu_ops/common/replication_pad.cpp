/**
* Copyright (c) 2021-2024 Intel Corporation
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