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
#include "backend/habana_device/HPUGuardImpl.h"
#include "generated/backend/index_reduce.h"

namespace habana {

FALLBACK_CHECK(
    IndexReduceFallbackCheck,
    int64_t dim,
    c10::string_view reduce,
    bool include_self) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  // Currently only small subset of features is supported by tpc kernel.
  if (include_self == true && reduce == "amax" && dim == 0 &&
      habana::HPUDeviceContext::get_device().type() != synDeviceGaudi) {
    return true;
  }
  return false;
};

std::shared_ptr<void> IndexReduceFillParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_IndexReduce::Params);
  params->axis = stack.at(1).toScalar().toInt();

  const auto mode = stack.at(4).toStringView();
  static const std::unordered_map<c10::string_view, IndexReduceMode_t>
      reduceModes = {
          {"amax", IndexReduceMode_t::INDEX_REDUCE_AMAX},
          {"amin", IndexReduceMode_t::INDEX_REDUCE_AMIN},
          {"mean", IndexReduceMode_t::INDEX_REDUCE_MEAN},
          {"prod", IndexReduceMode_t::INDEX_REDUCE_PROD},
      };

  auto it = reduceModes.find(mode);
  TORCH_CHECK(it != reduceModes.end(), "Unsupported reduce: ", mode)

  params->mode = it->second;
  params->include_self = stack.at(5).toBool();
  return params;
}

} // namespace habana
