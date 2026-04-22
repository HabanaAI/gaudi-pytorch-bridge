/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "generated/backend/grid_sampler_2d_backward.h"
#include "generated/backend/grid_sampler_3d.h"
#include "generated/backend/grid_sampler_3d_backward.h"
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

namespace habana {
OutputMetaDataVector GridSamplerMeta(const at::Stack& stack) {
  constexpr int SELF_POS = 0;
  constexpr int GRID_POS = 1;
  auto self = stack.at(SELF_POS).toTensor();
  auto grid = stack.at(GRID_POS).toTensor();
  // In the spatial (4-D) case, for input with shape (N,C,H^in,W^in) and grid
  // with shape (N,H^out,W^out,2), the output will have shape (N,C,H^out,W^out)
  // compute shape call from front-end
  // sizes are always in NCHW (irrespective of storage layout of physical
  // data)
  const bool is3d = self.dim() == 5;
  constexpr int N_SELF = 0;
  constexpr int C_SELF = 1;
  const int D_GRID = static_cast<int>(is3d);
  const int H_GRID = 1 + static_cast<int>(is3d);
  const int W_GRID = 2 + static_cast<int>(is3d);

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.dtype = self.scalar_type();
  meta.shape.reserve(4 + static_cast<int>(is3d));
  meta.shape = {self.sizes()[N_SELF], self.sizes()[C_SELF]};
  if (is3d) {
    meta.shape.emplace_back(grid.sizes()[D_GRID]);
  }
  meta.shape.emplace_back(grid.sizes()[H_GRID]);
  meta.shape.emplace_back(grid.sizes()[W_GRID]);
  return metaVec;
}

OutputMetaDataVector GridSamplerBwdMeta(const at::Stack& stack) {
  auto input = stack.at(1).toTensor();
  auto grid = stack.at(2).toTensor();

  auto CopyShape = [](OutputMetaData& meta, const at::Tensor& t) {
    meta.dtype = t.scalar_type();
    meta.shape = t.sizes().vec();
  };

  OutputMetaDataVector metaVec(2);
  CopyShape(metaVec[0], input);
  CopyShape(metaVec[1], grid);

  return metaVec;
}

static FillParamsT FillGridSamplerParamsCommon(
    const at::Stack& stack,
    int numInputTensors) {
  PARAMS_STUB(ns_GridSample::Params);
  const int INTERP_MODE_POS = numInputTensors;
  const int PAD_MODE_POS = numInputTensors + 1;
  const int ALIGN_COR_POS = numInputTensors + 2;
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
      HABANA_ASSERT(
          false,
          "Unsupported interpolation mode in grid_sampler op: ",
          interpolation_mode);
  }
  auto padding_mode = stack.at(PAD_MODE_POS).toInt();
  switch (padding_mode) {
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
      HABANA_ASSERT(
          false, "Unsupported padding mode in grid_sampler op: ", padding_mode);
  }
  auto align_corners = stack.at(ALIGN_COR_POS).toBool();
  params->alignCorners = align_corners;
  return paramsT;
}

FillParamsT FillGridSamplerParams(const at::Stack& stack) {
  return FillGridSamplerParamsCommon(stack, 2);
}

FillParamsT FillGridSamplerBwdParams(const at::Stack& stack) {
  return FillGridSamplerParamsCommon(stack, 3);
}
} // namespace habana
