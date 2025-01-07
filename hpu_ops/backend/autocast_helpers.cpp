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

#include <fstream>

#include "hpu_ops/autocast_helpers.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace at {
namespace autocast {

std::unordered_set<std::string> load_list(
    const std::string_view list_name,
    const std::unordered_set<std::string>& default_list) {
  auto path = std::getenv(list_name.data());
  if (path == nullptr) {
    PT_BRIDGE_DEBUG("Loaded default autocast list.")
    return default_list;
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
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    return arg.to(to_type);
  } else {
    return arg;
  }
}

} // namespace autocast
} // namespace at
