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

#include <ATen/native/GridSampler.h>
#include "generated/backend/grid_sampler_2d.h"
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

namespace habana {
OutputMetaDataVector GridSampler2dMeta(const at::Stack& stack) {
  constexpr int SELF_POS = 0;
  constexpr int GRID_POS = 1;
  auto self = stack.at(SELF_POS).toTensor();
  auto grid = stack.at(GRID_POS).toTensor();
  // In the spatial (4-D) case, for input with shape (N,C,H^in,W^in) and grid
  // with shape (N,H^out,W^out,2), the output will have shape (N,C,H^out,W^out)
  // compute shape call from front-end
  // sizes are always in NCHW (irrespective of storage layout of physical
  // data)
  constexpr int N_SELF = 0;
  constexpr int C_SELF = 1;
  constexpr int H_GRID = 1;
  constexpr int W_GRID = 2;

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = {
      self.sizes()[N_SELF],
      self.sizes()[C_SELF],
      grid.sizes()[H_GRID],
      grid.sizes()[W_GRID]};
  return {meta};
}

std::shared_ptr<void> FillGridSamplerParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_GridSample::Params);
  constexpr int INTERP_MODE_POS = 2;
  constexpr int PAD_MODE_POS = 3;
  constexpr int ALIGN_COR_POS = 4;
  auto interpolation_mode = stack.at(INTERP_MODE_POS).toInt();
  switch (interpolation_mode) {
    case static_cast<int64_t>(GridSamplerInterpolation::Bilinear):
      params->interp = GridSampleInterpolation_t::SAMPLE_BILINEAR;
      break;
    case static_cast<int64_t>(GridSamplerInterpolation::Nearest):
      params->interp = GridSampleInterpolation_t::SAMPLE_NEAREST;
      break;
    case static_cast<int64_t>(GridSamplerInterpolation::Bicubic):
      params->interp = GridSampleInterpolation_t::SAMPLE_CUBIC;
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported interpolation mode in grid_sampler op: ",
          interpolation_mode);
  }
  auto padding_mode = stack.at(PAD_MODE_POS).toInt();
  switch (interpolation_mode) {
    case static_cast<int64_t>(GridSamplerPadding::Zeros):
      params->pad = GridSamplePad_t::PAD_ZEROS;
      break;
    case static_cast<int64_t>(GridSamplerPadding::Border):
      params->pad = GridSamplePad_t::PAD_BORDER;
      break;
    case static_cast<int64_t>(GridSamplerPadding::Reflection):
      params->pad = GridSamplePad_t::PAD_REFLECTION;
      break;
    default:
      TORCH_CHECK(
          false, "Unsupported padding mode in grid_sampler op: ", padding_mode);
  }
  auto align_corners = stack.at(ALIGN_COR_POS).toBool();
  params->alignCorners = align_corners;
  return params;
}

} // namespace habana
