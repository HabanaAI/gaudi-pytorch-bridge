/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <fstream>

#include "hpu_ops/autocast_helpers.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace at {
namespace autocast {

inline constexpr std::string_view deprecated_flag(const std::string_view flag) {
  if (flag == AUTOCAST_LOWER_LIST) {
    return AUTOCAST_LOWER_LIST_DEPRECATED;
  } else if (flag == AUTOCAST_FP32_LIST) {
    return AUTOCAST_FP32_LIST_DEPRECATED;
  }
  return "";
}

std::unordered_set<std::string> load_list(
    const std::string_view list_name,
    const std::unordered_set<std::string>& default_list) {
  auto path = std::getenv(list_name.data());
  if (path == nullptr) {
    path = std::getenv(deprecated_flag(list_name).data());
    if (path) {
      PT_BRIDGE_WARN(
          AUTOCAST_LOWER_LIST_DEPRECATED,
          " and ",
          AUTOCAST_FP32_LIST_DEPRECATED,
          " are deprecated."
          " Use ",
          AUTOCAST_LOWER_LIST,
          " and ",
          AUTOCAST_FP32_LIST);
    } else {
      PT_BRIDGE_DEBUG("Loaded default autocast list.")
      return default_list;
    }
  }
  std::ifstream file(path);
  if (!file.is_open()) {
    PT_BRIDGE_WARN(
        "Failed to open file with ops to autocast: ",
        path,
        ". Default list loaded.");
    return default_list;
  }
  std::unordered_set<std::string> list;
  std::string line;
  std::string ops;
  while (getline(file, line)) {
    list.insert(line);
    ops += line + ", ";
  }
  PT_BRIDGE_DEBUG("Autocast ops loaded via ", list_name, ": ", ops);
  return list;
}

Tensor cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  // HPU in lazy mode doesn't benefit from cached casts. Potential optimization
  // are done in GC level. Moreover, it leaves persistent tensors from cast
  // operations, when HPU Graphs are used or .cpu() is called in the scope of
  // autocast. Since torch.autocast has caching enabled by default, to avoid the
  // risk of bad performance, cached casts are permanently disabled from
  // autocast on HPU.
  // TODO: SW-122613 Analyze impact of cached casts when graph mode in PT 2.0 is
  // introduced.
#if 0
  return cached_cast(to_type, arg, device_type);
#else
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    return arg.to(to_type);
  } else {
    return arg;
  }
#endif
}

} // namespace autocast
} // namespace at
