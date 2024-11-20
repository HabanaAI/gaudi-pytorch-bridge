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

#include <cstdint>
#include <string>
#include <vector>

#include <c10/macros/Export.h>
#include <c10/util/Optional.h>

#include "jit_fork/operator_upgraders/version_map.h"

namespace habana_torch {
namespace jit {

struct UpgraderRange {
  int min_version;
  int max_version;
};

// Given a list of upgrader entries for a single operator
// and the model version for that operator, find a valid
// upgrader.
TORCH_API c10::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

// Utility methods to find if the operator is up-to-date
// based on all registered upgraders for this operator.
// This can be different from the current server version
// because the implementation of this operator could have
// been consistent for many later version bumps.
TORCH_API bool isOpCurrentBasedOnUpgraderEntries(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

TORCH_API bool isOpSymbolCurrent(
    const std::string& name,
    size_t current_version);

// Returns the possible old schemas for the operator that
// doesn't exist anymore. This can be true for deprecated
// operators. Since name is always a symbol name, there
// can be multiple schemas for different overloads.
TORCH_API std::vector<std::string> loadPossibleHistoricOps(
    const std::string& name,
    c10::optional<size_t> version);

TORCH_API uint64_t getMaxOperatorVersion();

// Returns the list of min and max version numbers of the operators
// that an upgrader `x` support for all upgraders for op `foo`
TORCH_API std::vector<UpgraderRange> getUpgradersRangeForOp(
    const std::string& name);

} // namespace jit
} // namespace habana_torch
