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
