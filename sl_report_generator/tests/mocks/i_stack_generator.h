/**
 * Copyright (c) 2025 Intel Corporation
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

#pragma once

#include "sl_report_generator/src/stack_generator.h"

#include "mock.h"

namespace slrg::unit_tests {
class IStackGeneratorMock final : public slrg::IStackGenerator {
 public:
  std::vector<at::Stack> getStacks(at::ScalarType) override {
    return mock.mock_function<std::vector<at::Stack>>(__func__);
  }

  std::string getOpAndOverloadName() const override {
    return mock.mock_function<std::string>(__func__);
  }

  bool isBlacklistedPrecisionType(at::ScalarType) const override {
    return mock.mock_function<bool>(__func__);
  }

  bool isWhitelistedPrecisionType(at::ScalarType) const override {
    return mock.mock_function<bool>(__func__);
  }

  std::string DebugString() const override {
    return mock.mock_function<std::string>(__func__);
  }

  virtual ~IStackGeneratorMock() = default;

  Mock& configure() {
    return mock;
  }

  std::uint64_t getFunctionCallCounter(FunctionNameT function_name) {
    return mock.getFunctionCallCounter(function_name);
  }

 private:
  mutable Mock mock{"IStackGenerator"};
};
} // namespace slrg::unit_tests
