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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <c10/macros/Export.h>

namespace habana_torch {
namespace jit {

struct UpgraderEntry {
  int bumped_at_version;
  std::string upgrader_name;
  std::string old_schema;
};

// Toggle the behaviour of calculating version for the module.
// If this is true, we calculate solely based on upgraders
// If this is false, we calculate it based on historic per op version map
TORCH_API void calculate_package_version_based_on_upgraders(bool val);

TORCH_API bool get_version_calculator_flag();

TORCH_API const std::unordered_map<std::string, std::vector<UpgraderEntry>>&
get_operator_version_map();

TORCH_API void test_only_add_entry(
    const std::string& op_name,
    UpgraderEntry entry);

TORCH_API void test_only_remove_entry(const std::string& op_name);

TORCH_API void test_only_reset_flag();

} // namespace jit
} // namespace habana_torch
