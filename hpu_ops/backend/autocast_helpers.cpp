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

namespace at::autocast {

namespace {
std::filesystem::path get_top_level_directory(
    const std::filesystem::path& path) {
  const auto root_path = path.root_path();
  const auto top_level_dir = *std::next(std::begin(path));

  return root_path / top_level_dir;
}

// std::unordered_set does not support std::filesystem::path until gcc-11.4
struct path_hasher {
  auto operator()(const std::filesystem::path& path) const noexcept {
    return std::filesystem::hash_value(path);
  }
};

bool is_root_safe(const std::filesystem::path& path) {
  static const std::unordered_set<std::filesystem::path, path_hasher>
      banned_roots{
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
  bool is_valid{true};
  std::string reason;
};

ops_list_status validate_ops_list_path(
    const std::filesystem::path& path_to_list) {
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

std::string strip_namespace(const c10::OperatorName& op) {
  auto op_name = op.name;
  const auto separator_pos = op_name.find("::");
  op_name.erase(0, separator_pos + 2);
  return op_name;
}

auto get_registered_ops() {
  auto all_ops = c10::Dispatcher::singleton().getAllOpNames();
  const auto end_of_aten_ops = std::remove_if(
      std::begin(all_ops), std::end(all_ops), [](const auto& op) {
        const auto ns = op.getNamespace();
        return not ns.has_value() || ns.value() != "aten";
      });

  std::unordered_set<std::string> supported_op_list;
  std::transform(
      std::begin(all_ops),
      end_of_aten_ops,
      std::inserter(supported_op_list, std::begin(supported_op_list)),
      strip_namespace);

  return supported_op_list;
}

} // namespace

// This function use std::cerr for message reporting.
// Standard logging macros don't work during static initialization.
std::unordered_set<std::string> load_ops_list(
    const std::filesystem::path& path_to_list,
    const std::unordered_set<std::string>& default_list) {
  // Loading default empty path should silently return default list.
  if (path_to_list.empty()) {
    return default_list;
  }
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

  const auto supported_op_list = get_registered_ops();

  std::unordered_set<std::string> list;
  std::string op_name;
  std::string accepted_ops;
  for (auto op_index{1}; getline(file, op_name); ++op_index) {
    if (supported_op_list.find(op_name) == std::end(supported_op_list)) {
      std::cerr << "[WARNING][PT_BRIDGE][AUTOCAST] "
                << "Discarded unregistered op at line: " << op_index
                << std::endl;
      continue;
    }
    list.insert(op_name);
    accepted_ops += op_name + ", ";
  }

  std::cerr << "[DEBUG][PT_BRIDGE][AUTOCAST] "
            << "Autocast ops loaded via " << std::quoted(path_to_list.string())
            << ": " << accepted_ops << std::endl;

  return list;
}

} // namespace at::autocast
