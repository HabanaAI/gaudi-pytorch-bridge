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

/*******************************************************************************
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
 *******************************************************************************
 */
#pragma once

#include <cstdint>
#include <mutex>
#include <string>

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

// ****************************************************************************
// New style of env var declaration

#define GET_ENV_FLAG_NEW_READ_CACHE(e) \
  (env_flags::new_style::get_env_flag_new<env_flags::new_style::e>(#e, false))
#define GET_ENV_FLAG_NEW_SKIP_CACHE(e, c) \
  (env_flags::new_style::get_env_flag_new<env_flags::new_style::e>(#e, c))
#define GET_3RD_ARG(arg1, arg2, arg3, ...) arg3
#define GET_ENV_FLAG_NEW_ARG(...) \
  GET_3RD_ARG(                    \
      __VA_ARGS__, GET_ENV_FLAG_NEW_SKIP_CACHE, GET_ENV_FLAG_NEW_READ_CACHE)
#define GET_ENV_FLAG_NEW(...) GET_ENV_FLAG_NEW_ARG(__VA_ARGS__)(__VA_ARGS__)
#define SET_ENV_FLAG_NEW(e, v, o) \
  (env_flags::new_style::set_env_flag_new<env_flags::new_style::e>(#e, v, o))
#define UNSET_ENV_FLAG_NEW(e) \
  (env_flags::new_style::unset_env_flag_new<env_flags::new_style::e>(#e))
#define IS_ENV_FLAG_DEFINED_NEW(e) \
  (env_flags::new_style::is_defined_new<env_flags::new_style::e>(#e))
#define PARSE_ENV_FLAG_NEW(e, v)            \
  (env_flags::new_style::parse_env_by_type< \
      decltype(env_flags::new_style::e::actual_value)>(#e, v))

// ****************************************************************************

// default HCCL slicing for collectives. Update this in env to override it
// recommended values {AllReduce, ReduceScatter, AllGather : 128};
//                    {Reduce : 16}
#define DEFAULT_HCCL_SLICE_SIZE_MB 16

namespace env_flags {
// List of environment flags in the form:
// - name of the structure is identical with environment variable name
// - type of default value it the type the environment variable will be
// converted to
// - default value is assigned in case environment variable is undefined
// - min(), max() methods may be defined for range check

// Synapse-specific env var.
// Colon-separated list of tpc kernel libs to be loaded for GC

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
// macros in the top of this file that do symbol stringification
// automatically.

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

// ****************************************************************************
// New style of env var declaration

namespace new_style {

// Struct defination for string env variables
#define ENV_STRING_STRUCT_DEFINITION(NAME, DEFAULT_VAL)       \
  struct NAME {                                               \
    static bool is_cached;                                    \
    static bool is_defined;                                   \
    static std::string actual_value;                          \
    static constexpr const char* default_value = DEFAULT_VAL; \
  }

#define ENV_STRING_STRUCT_STATIC_DEFINITION(NAME) \
  bool NAME::is_cached{false};                    \
  bool NAME::is_defined{false};                   \
  std::string NAME::actual_value{};

// Struct defination for non-string env variables with numeric limits
#define ENV_STRUCT_DEFINITION(NAME, TYPE, DEFAULT_VAL) \
  struct NAME : public std::numeric_limits<TYPE> {     \
    static bool is_cached;                             \
    static bool is_defined;                            \
    static TYPE actual_value;                          \
    static constexpr TYPE default_value = DEFAULT_VAL; \
  }

#define ENV_STRUCT_STATIC_DEFINITION(NAME, TYPE) \
  bool NAME::is_cached{false};                   \
  bool NAME::is_defined{false};                  \
  TYPE NAME::actual_value{};

// Method for string env variables
const char* getenv_by_type_new(
    const char* name,
    const bool& skip_cache,
    bool& is_cached,
    bool& is_defined,
    std::string& act_val,
    const char* def_val);

// Method for bool env variables to handle "true"/"false" and 1/0
bool getenv_by_type_new(
    const char* name,
    const bool& skip_cache,
    bool& is_cached,
    bool& is_defined,
    bool& act_val,
    const bool def_val,
    const bool min_val,
    const bool max_val);

// Template method(s) for non-string and non-bool env variables
template <class T>
T getenv_by_type_new(
    const char* name,
    const bool& skip_cache,
    bool& is_cached,
    bool& is_defined,
    T& act_val,
    const T def_val,
    const T min_val,
    const T max_val);

template <class T>
T parse_env_by_type(const char* name, const char* value);

/*
 * Template method(s) for setting env variables
 *
 * Template arguments for Env variables data types are same i.e.
 * A == N (actual value data type == default value/new value data type)
 * Except for string Env variables where A (actual value)
 * is of type std::string and N (new value) is of type const char*
 */
template <class A, class N>
void setenv_by_type_new(
    const char* name,
    bool& is_cached,
    bool& is_defined,
    A& act_val,
    const N new_val,
    int overwrite) {
  (void)name;
  if (!is_defined || overwrite) {
    act_val = new_val;
    is_cached = true;
    is_defined = true;
  }
}

template <class E>
typename std::
    enable_if<!has_min_max_methods<E>::value, decltype(E::default_value)>::type
    getenv_E_new(const char* name, const bool& skip_cache) {
  return getenv_by_type_new(
      name,
      skip_cache,
      E::is_cached,
      E::is_defined,
      E::actual_value,
      E::default_value);
}

template <class E>
typename std::
    enable_if<has_min_max_methods<E>::value, decltype(E::default_value)>::type
    getenv_E_new(const char* name, const bool& skip_cache) {
  return getenv_by_type_new(
      name,
      skip_cache,
      E::is_cached,
      E::is_defined,
      E::actual_value,
      E::default_value,
      E::min(),
      E::max());
}

template <class E>
void setenv_E_new(
    const char* name,
    const decltype(E::default_value) new_val,
    int overwrite) {
  setenv_by_type_new(
      name, E::is_cached, E::is_defined, E::actual_value, new_val, overwrite);
}

template <class E>
decltype(E::default_value) get_env_flag_new(
    const char* name,
    const bool& skip_cache) {
  return getenv_E_new<E>(name, skip_cache);
}

// setenv mode, If overwrite is 'non zero' value. It overwrites existing env
// value if defined
template <class E>
void set_env_flag_new(
    const char* name,
    const decltype(E::default_value) val,
    int overwrite) {
  setenv_E_new<E>(name, val, overwrite);
}

template <class E>
void unset_env_flag_new(const char* name) {
  (void)name;
  E::is_cached = false;
  E::is_defined = false;
}

template <class E>
void update_is_defined(const char* name) {
  const char* envstrp = getenv(name);
  E::is_defined = envstrp && *envstrp;
}

template <class E>
bool is_defined_new(const char* name) {
  static std::once_flag flag;
  if (!E::is_defined)
    std::call_once(flag, update_is_defined<E>, name);
  return E::is_defined;
}

} // namespace new_style

// ****************************************************************************

} // namespace env_flags
