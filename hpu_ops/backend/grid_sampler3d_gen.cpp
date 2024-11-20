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
#include "generated/backend/grid_sampler_3d.h"

namespace habana {
OutputMetaDataVector GridSampler3dMeta(const at::Stack& stack) {
  constexpr auto SELF_POS = 0;
  constexpr auto GRID_POS = 1;
  auto self = stack.at(SELF_POS).toTensor();
  auto grid = stack.at(GRID_POS).toTensor();
  constexpr auto N_SELF = 0;
  constexpr auto C_SELF = 1;
  constexpr auto D_GRID = 1;
  constexpr auto H_GRID = 2;
  constexpr auto W_GRID = 3;

  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = {
      self.sizes()[N_SELF],
      self.sizes()[C_SELF],
      grid.sizes()[D_GRID],
      grid.sizes()[H_GRID],
      grid.sizes()[W_GRID]};
  return {meta};
}
} // namespace habana
