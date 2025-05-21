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

#include "hpu_ops/op_validator.h"

#include "mock.h"

namespace slrg::unit_tests {
class CheckNodeWithSharedLayerValidatorMock final {
 public:
  bool Validate(
      const at::Stack&,
      [[maybe_unused]] bool is_dynamic = false,
      [[maybe_unused]] bool check_st_h2d = false,
      [[maybe_unused]] const habana::SharedMetaVector& meta = {},
      [[maybe_unused]] std::optional<SharedLayer::DeviceId> device_stub =
          std::nullopt) {
    return mock.mock_function<bool>(__func__);
  }

  bool ValidateCustom(
      const at::Stack&,
      [[maybe_unused]] bool is_dynamic = false,
      [[maybe_unused]] bool check_st_h2d = false,
      [[maybe_unused]] std::optional<SharedLayer::DeviceId> device_stub =
          std::nullopt) {
    return mock.mock_function<bool>(__func__);
  }

  Mock& configure() {
    return mock;
  }

  std::uint64_t getFunctionCallCounter(FunctionNameT function_name) {
    return mock.getFunctionCallCounter(function_name);
  }

 private:
  mutable Mock mock{"CheckNodeWithSharedLayerValidator"};
};
} // namespace slrg::unit_tests
