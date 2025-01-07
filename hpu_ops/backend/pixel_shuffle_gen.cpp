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
#include <perf_lib_layer_params.h>
#include "generated/backend/pixel_shuffle.h"

namespace habana {

OutputMetaDataVector PixelShuffleMeta(const at::Stack& stack) {
  auto input = stack.at(0).toTensor();
  const auto upscaleFactor = stack.at(1).toInt();
  auto shape = input.sizes().vec();
  auto rank = shape.size();
  shape[rank - 3] /= upscaleFactor * upscaleFactor;
  shape[rank - 2] *= upscaleFactor;
  shape[rank - 1] *= upscaleFactor;

  OutputMetaData meta;
  meta.shape = shape;
  meta.dtype = input.scalar_type();

  return {meta};
}

std::shared_ptr<void> FillPixelShuffleParams(
    const at::Stack& stack,
    size_t& size) {
  const auto upscaleFactor = stack.at(1).toInt();
  PARAMS_STUB(ns_PixelShuffleKernel::Params);
  params->upscale_factor = upscaleFactor;
  return params;
}

} // namespace habana
