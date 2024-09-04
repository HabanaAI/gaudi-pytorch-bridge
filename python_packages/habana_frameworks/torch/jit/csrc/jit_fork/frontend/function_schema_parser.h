/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <string>
#include <variant>

namespace habana_torch {
namespace jit {

TORCH_API std::variant<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName);
TORCH_API c10::FunctionSchema parseSchema(const std::string& schema);
TORCH_API c10::OperatorName parseName(const std::string& name);

} // namespace jit
} // namespace habana_torch
