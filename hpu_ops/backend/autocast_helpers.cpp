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
#include <iomanip>
#include <iostream>
#include <sstream>

#include "hpu_ops/autocast_helpers.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace at {
namespace autocast {

namespace {
std::filesystem::path get_top_level_directory(const std::filesystem::path& path) {
  const auto root_path = path.root_path();
  const auto top_level_dir = *std::next(std::begin(path));

  return root_path / top_level_dir;
}

bool is_root_safe(const std::filesystem::path& path) {
  static const std::unordered_set<std::filesystem::path> banned_roots{
      "/bin",
      "/etc",
      "/usr",
      "/var",
      "/boot",
      "/dev",
      "/lib",
      "/lost+found",
      "/proc",
      "/run",
      "/sbin",
      "/srv",
      "/sys",
      "/root"};
  const auto absolute_path = std::filesystem::absolute(path);
  const auto canonical_path = std::filesystem::canonical(absolute_path);
  const auto top_level_dir = get_top_level_directory(canonical_path);

  return banned_roots.find(top_level_dir) == std::end(banned_roots);
}

struct ops_list_status {
  bool is_valid = true;
  std::string reason = "";
};

const ops_list_status validate_ops_list_path(
    const std::filesystem::path& path_to_list) {
  if (path_to_list.empty())
    return {false, "Empty path."};

  if (not std::filesystem::exists(path_to_list)) {
    std::stringstream reason;
    reason << "File(" << std::quoted(path_to_list.string())
           << ") does not exist.";
    return {false, reason.str()};
  }

  if (std::filesystem::is_directory(path_to_list)) {
    std::stringstream reason;
    reason << "Path(" << std::quoted(path_to_list.string())
           << ") is a directory";
    return {false, reason.str()};
  }

  if (not is_root_safe(path_to_list)) {
    std::stringstream reason;
    reason << "Used unsafe file location(" << std::quoted(path_to_list.string())
           << ").";
    return {false, reason.str()};
  }

  return {};
}
} // namespace

// This function use std::cerr for message reporting.
// Standard logging macros doesn't work during static initialization.
std::unordered_set<std::string> load_ops_list(
    const std::filesystem::path& path_to_list,
    const std::unordered_set<std::string>& default_list) {
  const auto verification_status = validate_ops_list_path(path_to_list);
  if (not verification_status.is_valid) {
    std::cerr << "[WARNING][PT_BRIDGE][AUTOCAST] "
              << "Verification of ops list to autocast failed: "
              << verification_status.reason << " Loading default ops list."
              << std::endl;
    return default_list;
  }

  std::ifstream file(path_to_list);
  if (not file.is_open()) {
    std::cerr << "[WARNING][PT_BRIDGE][AUTOCAST] "
              << "Opening of ops list(" << std::quoted(path_to_list.string())
              << ") to autocast failed."
              << " Loading default list." << std::endl;
    return default_list;
  }

  std::unordered_set<std::string> list;
  std::string line;
  std::string ops;
  while (getline(file, line)) {
    list.insert(line);
    ops += line + ", ";
  }

  std::cerr << "[DEBUG][PT_BRIDGE][AUTOCAST] "
            << "Autocast ops loaded via " << std::quoted(path_to_list.string())
            << ": " << ops << std::endl;

  return list;
}

} // namespace autocast
} // namespace at
