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

/* Enum-like class that also is equivalent to 'true' when op is supported.
 * Otherwise it carries a more detailed reason why it isn't.
 */
class OpSupportLevel {
 public:
  enum class Value {
    supported,
    placed_on_cpu,
    unsupported_dtype,
    unsupported_args, // any other issue with argument that is not wrong dtype
    unsupported_rank,
    unsupported
  };
  explicit operator bool() {
    return value == Value::supported;
  }
  OpSupportLevel& operator=(OpSupportLevel::Value v) {
    value = v;
    return *this;
  }
  OpSupportLevel(Value v) : value{v} {} // allow implicit
  OpSupportLevel() = delete;

  Value value;
};

inline bool operator==(OpSupportLevel a, OpSupportLevel b) {
  return a.value == b.value;
}

inline bool operator!=(OpSupportLevel a, OpSupportLevel b) {
  return !(a == b);
}
