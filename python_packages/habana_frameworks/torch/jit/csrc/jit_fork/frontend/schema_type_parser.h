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

#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <c10/util/FunctionRef.h>

#include "jit_fork/frontend/lexer.h"
#include "jit_fork/ir/type_wrapper.h"

namespace habana_torch {
namespace jit {

using TypePtr = c10::TypePtr;

struct TORCH_API SchemaTypeParser {
  TypeWrapper parseBaseType();
  c10::optional<c10::AliasInfo> parseAliasAnnotation();
  std::pair<TypeWrapper, c10::optional<c10::AliasInfo>> parseType();
  std::tuple<
      /*fake*/ TypeWrapper,
      /*real*/ TypeWrapper,
      c10::optional<c10::AliasInfo>>
  parseFakeAndRealType();
  c10::optional<at::ScalarType> parseTensorDType(const std::string& dtype);
  TypeWrapper parseRefinedTensor();

  SchemaTypeParser(Lexer& L, bool parse_complete_tensor_types)
      : complete_tensor_types(parse_complete_tensor_types), L(L) {}

 private:
  c10::optional<bool> tryToParseRequiresGrad();
  c10::optional<c10::Device> tryToParseDeviceType();
  void parseList(
      int begin,
      int sep,
      int end,
      c10::function_ref<void()> callback);
  std::string parseUntil(int kind);
  std::string parseUntil(
      const std::vector<int>& kinds,
      const std::vector<std::pair<int, int>>& following_kinds =
          std::vector<std::pair<int, int>>());

  bool complete_tensor_types;
  Lexer& L;
  size_t next_id = 0;
};

} // namespace jit
} // namespace habana_torch
