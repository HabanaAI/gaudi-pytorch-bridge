/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
