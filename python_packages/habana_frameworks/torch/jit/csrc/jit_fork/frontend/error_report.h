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

#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "jit_fork/frontend/source_range.h"

namespace habana_torch {
namespace jit {

struct Token;

struct Call {
  std::string fn_name;
  SourceRange caller_range;
};

struct ErrorReport : public std::exception {
  ErrorReport(const ErrorReport& e);

  explicit ErrorReport(SourceRange r);

  const char* what() const noexcept override;

  struct CallStack {
    // These functions are used to report why a function was being compiled
    // (i.e. what was the call stack of user functions at compilation time that
    // led to this error)
    CallStack(const std::string& name, const SourceRange& range);
    ~CallStack();

    // Change the range that is relevant for the current function (i.e. after
    // each successful expression compilation, change it to the next expression)
    static void update_pending_range(const SourceRange& range);
  };

  static std::string current_call_stack();

 private:
  template <typename T>
  friend const ErrorReport& operator<<(const ErrorReport& e, const T& t);

  mutable std::stringstream ss;
  OwnedSourceRange context;
  mutable std::string the_message;
  std::vector<Call> error_stack;
};

template <typename T>
const ErrorReport& operator<<(const ErrorReport& e, const T& t) {
  e.ss << t;
  return e;
}

} // namespace jit
} // namespace habana_torch
