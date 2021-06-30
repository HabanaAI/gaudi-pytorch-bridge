/*
 * INTEL CONFIDENTIAL
 * Copyright 2018-2020 Intel Corporation.
 *
 * This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 * This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in
 * the License.
 */
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

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

// Macros are necessary for creating string from symbol
// And we need both, string for reading environment variable
// and symbol for handling it

// Reads environment variable and converts it to type defined by
// E::default_value
#define GET_ENV_FLAG(e) (env_flags::get_env_flag<env_flags::e>(#e))

// If environment variable is defined converts it to type defined by
// E::default_value and overwrittes v with this value
#define ENV_FLAG_OVERRIDE(e, v) \
  (env_flags::env_flag_override<env_flags::e>(#e, v))

// As above but calls f(user_value) instead of direct overwritting
#define ENV_FLAG_OVERRIDE_CUSTOM(e, f) \
  (env_flags::env_flag_override_custom<env_flags::e>(#e, f))

// Returns true if environment variable is defined false otherwise
#define IS_ENV_FLAG_DEFINED(e) (env_flags::is_defined<env_flags::e>(#e))

namespace env_flags {
// List of environment flags in the form:
// - name of the structure is identical with environment variable name
// - type of default value it the type the environment variable will be
// converted to
// - default value is assigned in case environment variable is undefined
// - min(), max() methods may be defined for range check

struct PT_ENABLE_HABANA_CACHING {
  static constexpr bool default_value = true;
};

struct PT_ENABLE_HABANA_STREAMASYNC {
  static constexpr bool default_value = true;
};

struct PT_ENABLE_HOST_MEMORY_CACHE {
  static constexpr bool default_value = true;
};

struct PT_ENABLE_HCL_SAME_ADDRESS_RESOLUTION {
  static constexpr bool default_value = false;
};

struct PT_ENABLE_HCL_STREAM {
  static constexpr bool default_value = true;
};

struct PT_ENABLE_DYNAMIC_WB {
  static constexpr bool default_value = true;
};

struct PT_HABANA_MAX_DMA_COPY_RETRY_COUNT
    : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 1000;
};

struct PT_HABANA_DMA_COPY_RETRY_DELAY : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 10;
};

struct PT_HABANA_POOL_SIZE : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = 24;
};

struct PT_HPU_POOL_STRATEGY : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 3;
};

struct PT_HABANA_MEM_LOG_LEVEL : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 0;
};

struct PT_HABANA_MAX_RECIPE_HIT_COUNT : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 0;
};

struct PT_HABANA_MEM_LOG_FILENAME {
  static constexpr const char* default_value = "habana_log.livealloc.log";
};

struct PT_ENABLE_SYNC_OUTPUT_HOST {
  static constexpr bool default_value = true;
};

struct PT_HPU_ENABLE_SYNC_OUTPUT_HOST {
  static constexpr bool default_value = true;
};

struct PT_USE_HCL_SYNC {
  static constexpr bool default_value = false;
};

struct PT_HPU_USE_HCL_SYNC {
  static constexpr bool default_value = false;
};

struct PT_USE_HCL_OPTS {
  static constexpr bool default_value = false;
};

struct PT_HPU_LOWER_AS_STRIDED {
  static constexpr bool default_value = true;
};

struct PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES {
  static constexpr bool default_value = false;
};

struct PT_HPU_ENABLE_INTERMEDIATE_TENSOR_RELEASE {
  static constexpr bool default_value = false;
};

struct PT_HPU_PRINT_BACKTRACE_ON_SIGNAL {
  static constexpr bool default_value = true;
};

// Overloads for different type of default value

template <class T>
using RT = std::pair<T, bool>;

template <class T>
RT<T> getenv_by_type(
    const char* name,
    const T def_val,
    const T min_val,
    const T max_val);

template <class T>
RT<T> getenv_by_type(const char* name, const T def_val);

// Overloads for beging derived from std::numeric_limits or not

template <class E>
struct has_min_max_methods {
  using value_type = decltype(E::default_value);
  using value_type_decay = typename std::decay<value_type>::type;
  using base_class = std::numeric_limits<value_type_decay>;
  static constexpr bool value = std::is_base_of<base_class, E>::value;
};

template <class E>
typename std::enable_if<
    has_min_max_methods<E>::value,
    RT<decltype(E::default_value)>>::type
getenv_by_E(const char* name) {
  return getenv_by_type(name, E::default_value, E::min(), E::max());
}

template <class E>
typename std::enable_if<
    !has_min_max_methods<E>::value,
    RT<decltype(E::default_value)>>::type
getenv_by_E(const char* name) {
  return getenv_by_type(name, E::default_value);
}

// Utility functions. It is more convenient to use them indirectly through
// macros in the top of this file that do symbol stringification automatically.

template <class E>
decltype(E::default_value) get_env_flag(const char* name) {
  return getenv_by_E<E>(name).first;
}

template <class E, class F>
void env_flag_override_custom(const char* name, F update) {
  auto pair = getenv_by_E<E>(name);
  if (pair.second) {
    update(pair.first);
  }
}

template <class E, class T>
void env_flag_override(const char* name, T& value) {
  env_flag_override_custom<E>(
      name, [&value](T new_value) { value = new_value; });
}

template <class E>
bool is_defined(const char* name) {
  bool ret = false;
  env_flag_override_custom<E>(
      name, [&ret](decltype(E::default_value)) { ret = true; });
  return ret;
}

} // namespace env_flags
