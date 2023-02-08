/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
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
