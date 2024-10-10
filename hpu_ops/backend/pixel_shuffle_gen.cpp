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
#include <perf_lib_layer_params.h>
#include "generated/backend/pixel_shuffle.h"

namespace habana {

namespace sh = synapse_helpers;

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
