/**
 * Copyright (c) 2021-2025 Intel Corporation
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

std::unordered_set<std::string> load_ops_list(
    const std::filesystem::path& path_to_list,
    const std::unordered_set<std::string>& default_list) {
  if (path_to_list.empty()) {
    PT_BRIDGE_DEBUG("Loaded default autocast list.")
    return default_list;
  }

  std::ifstream file(path_to_list);
  if (not file.is_open()) {
    PT_BRIDGE_WARN(
        "Failed to open file with ops to autocast: ",
        path_to_list,
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

  PT_BRIDGE_DEBUG("Autocast ops loaded via ", path_to_list, ": ", ops);
  return list;
}

} // namespace autocast
} // namespace at
