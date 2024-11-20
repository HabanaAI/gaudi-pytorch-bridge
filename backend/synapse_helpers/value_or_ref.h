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
#pragma once

#include <absl/types/variant.h>
#include <functional>
#include <utility>

namespace synapse_helpers {

template <typename T>
class value_or_ref {
 public:
  using underlying_type = absl::variant<T, std::reference_wrapper<T>>;

  value_or_ref(T& input) : value_(std::ref(input)) {}
  value_or_ref(std::reference_wrapper<T> input) : value_(input) {}
  value_or_ref(T&& input) : value_(std::move(input)) {}

  operator T&() {
    return ref();
  }
  operator const T&() const {
    return absl::visit(value_ref_caster{}, value_);
  }
  T& ref() {
    return absl::visit(value_ref_caster{}, value_);
  }

 private:
  struct value_ref_caster {
    template <typename U>
    T& operator()(U& value) {
      return value;
    }
    template <typename U>
    const T& operator()(const U& value) {
      return value;
    }
  };

  underlying_type value_;
};

} // namespace synapse_helpers
