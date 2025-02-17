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
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "jit_fork/operator_upgraders/utils.h"

#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <c10/util/Optional.h>
#include <caffe2/serialize/versions.h>

#include "jit_fork/operator_upgraders/version_map.h"

namespace habana_torch {
namespace jit {

c10::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version) {
  // we want to find the entry which satisfies following two conditions:
  //    1. the version entry must be greater than current_version
  //    2. Among the version entries, we need to see if the current version
  //       is in the upgrader name range
  auto pos = std::find_if(
      upgraders_for_schema.begin(),
      upgraders_for_schema.end(),
      [current_version](const UpgraderEntry& entry) {
        return entry.bumped_at_version > static_cast<int>(current_version);
      });

  if (pos != upgraders_for_schema.end()) {
    return *pos;
  }
  return c10::nullopt;
}

bool isOpCurrentBasedOnUpgraderEntries(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version) {
  auto latest_update =
      upgraders_for_schema[upgraders_for_schema.size() - 1].bumped_at_version;
  if (latest_update > static_cast<int>(current_version)) {
    return false;
  }
  return true;
}

bool isOpSymbolCurrent(const std::string& name, size_t current_version) {
  auto it = get_operator_version_map().find(name);
  if (it != get_operator_version_map().end()) {
    return isOpCurrentBasedOnUpgraderEntries(it->second, current_version);
  }
  return true;
}

std::vector<std::string> loadPossibleHistoricOps(
    const std::string& name,
    c10::optional<size_t> version) {
  std::vector<std::string> possibleSchemas;

  if (!version.has_value()) {
    return possibleSchemas;
  }

  for (const auto& entry : get_operator_version_map()) {
    auto old_symbol_name = entry.first;
    // strip off the overload name, if exist
    auto base_name = old_symbol_name.substr(0, old_symbol_name.find('.'));
    if (base_name == name) {
      auto possibleUpgrader = findUpgrader(entry.second, version.value());
      if (possibleUpgrader.has_value()) {
        possibleSchemas.push_back(possibleUpgrader.value().old_schema);
      }
    }
  }

  return possibleSchemas;
}

uint64_t getMaxOperatorVersion() {
  return caffe2::serialize::kProducedFileFormatVersion;
}

std::vector<UpgraderRange> getUpgradersRangeForOp(const std::string& name) {
  std::vector<UpgraderRange> output;
  auto it = get_operator_version_map().find(name);
  if (it == get_operator_version_map().end()) {
    return output;
  }

  output.reserve(it->second.size());
  int cur_min = 0;
  for (const auto& entry : it->second) {
    int cur_max = entry.bumped_at_version - 1;
    output.emplace_back(UpgraderRange{cur_min, cur_max});
    cur_min = entry.bumped_at_version;
  }
  return output;
}

} // namespace jit
} // namespace habana_torch
