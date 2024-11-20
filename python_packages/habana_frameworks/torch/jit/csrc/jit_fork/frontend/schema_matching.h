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

#include <ATen/core/function_schema.h>
#include <c10/macros/Export.h>

#include "jit_fork/ir/ir.h"
#include "jit_fork/ir/named_value.h"

namespace habana_torch {
namespace jit {

// Try to match a list of inputs and keyword 'attributes' to this
// schema. Return the flat list of positional inputs to the call or
// `c10::nullopt` on failure (`failure_messages` contains a good error
// report in this case)

struct MatchedSchema {
  std::vector<Value*> inputs;
  std::vector<TypePtr> return_types;
  c10::OptNameList return_field_names;
  std::string schema_name;
};

TORCH_API bool isBlockListedSchema(const FunctionSchema& schema);

TORCH_API MatchedSchema matchSchema(
    const ::c10::FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

TORCH_API std::pair<size_t, MatchedSchema> matchSchemas(
    const std::vector<const ::c10::FunctionSchema*>& schemas,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt,
    bool render_errors = false);

TORCH_API bool convertibleToList(
    const TypePtr& type,
    const TypePtr& list_type_);

TORCH_API std::string getFullSchemaName(const ::c10::FunctionSchema& schema);

TORCH_API Value* emitBuiltinCall(
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    const c10::optional<NamedValue>& self = c10::nullopt);

TORCH_API c10::optional<size_t> findInputWithName(
    const std::string& name,
    at::ArrayRef<NamedValue> kwargs,
    bool is_aten = false);

// applies implicit conversion from value trying to turn it into type
// concrete_type it succeeds if the return_value->isSubtypeOf(concrete_type)
TORCH_API Value* tryConvertToType(
    const SourceRange& loc,
    Graph& graph,
    const TypePtr& concrete_type,
    Value* value,
    bool allow_conversions);

} // namespace jit
} // namespace habana_torch
