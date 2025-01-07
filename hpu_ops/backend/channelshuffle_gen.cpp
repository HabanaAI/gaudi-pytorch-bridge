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

#include "generated/backend/channel_shuffle.h"

namespace habana {

OutputMetaDataVector ChannelShuffleMeta(const at::Stack& stack) {
  auto input = stack.at(0).toTensor();
  TORCH_CHECK(
      input.dim() > 2,
      "Channel shuffle expects input with dim > 2, but got ",
      input.dim());

  const int groups = stack.at(1).toInt();
  TORCH_CHECK(
      groups > 0,
      "Channel shuffle expects number of groups to be positive, but got ",
      groups);

  const auto shape = input.sizes().vec();
  const auto inputChannels = shape[1];
  TORCH_CHECK(
      (inputChannels % groups == 0),
      "Channel shuffle expects number of channels to be divisible by groups");

  OutputMetaData meta;
  meta.shape = shape;
  meta.dtype = input.scalar_type();

  return {meta};
}

std::shared_ptr<void> FillChannelShuffleParams(
    const at::Stack& stack,
    size_t& size) {
  const auto groups = stack.at(1).toInt();
  PARAMS_STUB(ns_ChannelShuffle::Params);
  params->groups = groups;
  return params;
}

} // namespace habana
