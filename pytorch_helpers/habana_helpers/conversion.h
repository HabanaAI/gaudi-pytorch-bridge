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

#include <cstdint>
#include <limits>
#include <string_view>

template <typename To, typename From>
To safe_convert(
    From value,
    const std::string_view file_name = __builtin_FILE(),
    const int line_number = __builtin_LINE(),
    const std::string_view function_name = __builtin_FUNCTION()) {
  static_assert(std::is_integral_v<From> && std::is_integral_v<To>);
  if constexpr (std::is_same_v<From, To>) {
    return value; // No conversion needed
  }

  // Single unified check using constexpr conditions
  constexpr bool is_same_sign = std::is_signed_v<From> == std::is_signed_v<To>;
  using larger_type = std::conditional_t<(sizeof(From) > sizeof(To)), From, To>;
  constexpr bool is_larger_target = sizeof(To) > sizeof(From);
  constexpr bool is_unsigned_to_signed =
      std::is_unsigned_v<From> && std::is_signed_v<To>;

  // For signed to unsigned, check non-negative; for others, check range
  constexpr auto to_min = std::numeric_limits<To>::min();
  constexpr auto to_max = std::numeric_limits<To>::max();

  const auto is_same_sign_and_in_range = is_same_sign &&
      static_cast<larger_type>(value) >= static_cast<larger_type>(to_min) &&
      static_cast<larger_type>(value) <= static_cast<larger_type>(to_max);

  const auto is_unsigned_to_signed_and_in_range =
      is_unsigned_to_signed && value <= static_cast<From>(to_max);

  const auto is_not_unsigned_to_signed_and_in_range = !is_unsigned_to_signed &&
      value >= static_cast<From>(to_min) &&
      static_cast<uint64_t>(value) <= static_cast<uint64_t>(to_max);

  HABANA_ASSERT(
      is_larger_target || is_same_sign_and_in_range ||
          is_unsigned_to_signed_and_in_range ||
          is_not_unsigned_to_signed_and_in_range,
      "Value out of range for conversion in ",
      file_name,
      ":",
      line_number,
      " (",
      function_name,
      ") - value: ",
      value);

  return static_cast<To>(value);
}
