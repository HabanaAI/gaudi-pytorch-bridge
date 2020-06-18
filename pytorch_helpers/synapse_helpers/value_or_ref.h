/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
