/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <algorithm>
#include <climits>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_set>

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

#define PP_NARG(...) PP_NARG_(__VA_ARGS__, PP_RSEQ_N())
#define PP_NARG_(...) PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N(_1, _2, _3, _4, _5, _6, _7, N, ...) N
#define PP_RSEQ_N() 7, 6, 5, 4, 3, 2, 1, 0

#define COUNT_ARGS(...) PP_NARG(__VA_ARGS__)

#define ENV_STRUCT_DEFINITION_SELECTOR(count) ENV_STRUCT_DEFINITION_##count
#define ENV_STRUCT_DEFINITION_DISPATCHER(count) \
  ENV_STRUCT_DEFINITION_SELECTOR(count)

#define ENV_STRUCT_DEFINITION(...) \
  ENV_STRUCT_DEFINITION_DISPATCHER(COUNT_ARGS(__VA_ARGS__))(__VA_ARGS__)

#define ENV_STRING_STRUCT_DEFINITION_SELECTOR(count) \
  ENV_STRING_STRUCT_DEFINITION_##count
#define ENV_STRING_STRUCT_DEFINITION_DISPATCHER(count) \
  ENV_STRING_STRUCT_DEFINITION_SELECTOR(count)

#define ENV_STRING_STRUCT_DEFINITION(...) \
  ENV_STRING_STRUCT_DEFINITION_DISPATCHER(COUNT_ARGS(__VA_ARGS__))(__VA_ARGS__)

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
  using value_type_decay = typename std::decay_t<value_type>;
  using base_class = std::numeric_limits<value_type_decay>;
  static constexpr bool value = std::is_base_of_v<base_class, E>;
};

template <class E>
typename std::
    enable_if_t<has_min_max_methods<E>::value, RT<decltype(E::default_value)>>
    getenv_by_E(const char* name) {
  return getenv_by_type(name, E::default_value, E::min(), E::max());
}

template <class E>
typename std::
    enable_if_t<!has_min_max_methods<E>::value, RT<decltype(E::default_value)>>
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

// Constraint types
constexpr const char* ENUM_CONSTRAINT_TYPE = "enum";
constexpr const char* RANGE_CONSTRAINT_TYPE = "range";
constexpr const char* FILEPATH_CONSTRAINT_TYPE = "filepath";
constexpr const char* LIST_CONSTRAINT_TYPE = "list";
constexpr const char* CUSTOM_CONSTRAINT_TYPE = "custom";

// Flag statuses
constexpr const char* FLAG_STATUS_DEPRECATED = "deprecated";
constexpr const char* FLAG_STATUS_OBSOLETE = "obsolete";

// Regular expression pattern to match potentially insecure shell characters.
const std::string insecure_pattern_str = R"([`$&;|<>\\])";
const std::regex insecure_pattern(insecure_pattern_str);
// Regular expression pattern to match potentially unsafe characters in input
// strings
const std::string unsafe_pattern_str = R"([\s"'\\<>|&;$%*?\[\]\{\}^~`])";
const std::regex unsafe_pattern(unsafe_pattern_str);

constexpr const char* CONSTRAINTS_EMPTY = "";
constexpr const char* CONSTRAINTS_TRUE = "true";
constexpr const char* CONSTRAINTS_FALSE = "false";
constexpr const char* CONSTRAINTS_CREATE_TRUE = "create=true";
constexpr char CONSTRAINTS_SPLIT_COMMA = ',';
constexpr char CONSTRAINTS_SPLIT_PIPE = '|';
constexpr char CONSTRAINTS_SPLIT_EQUAL = '=';
constexpr char CONSTRAINTS_SPLIT_AND = '&';

// Struct defination for string env variables
#define ENV_STRING_STRUCT_DEFINITION_5(                          \
    NAME, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CONSTRAINS) \
  ENV_STRING_STRUCT_DEFINITION_FULL(                             \
      NAME, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CONSTRAINS)

#define ENV_STRING_STRUCT_DEFINITION_3(NAME, DEFAULT_VAL, FLAG_STATUS) \
  ENV_STRING_STRUCT_DEFINITION_FULL(                                   \
      NAME, DEFAULT_VAL, FLAG_STATUS, nullptr, nullptr)

#define ENV_STRING_STRUCT_DEFINITION_WITH_CUSTOM(                           \
    NAME, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CHECK_FUNC)            \
  struct NAME {                                                             \
    static bool is_cached;                                                  \
    static bool is_defined;                                                 \
    static constexpr const char* constrains_type = CONSTRAINS_TYPE;         \
    static std::string actual_value;                                        \
    static constexpr const char* default_value = DEFAULT_VAL;               \
    static constexpr const char* flag_status = FLAG_STATUS;                 \
    static constexpr const char* constrains = nullptr;                      \
    static constexpr bool (*check_func)(const std::string&, std::string&) = \
        CHECK_FUNC;                                                         \
  }

#define ENV_STRING_STRUCT_DEFINITION_FULL(                                  \
    NAME, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CONSTRAINS)            \
  struct NAME {                                                             \
    static bool is_cached;                                                  \
    static bool is_defined;                                                 \
    static constexpr const char* constrains_type = CONSTRAINS_TYPE;         \
    static std::string actual_value;                                        \
    static constexpr const char* default_value = DEFAULT_VAL;               \
    static constexpr const char* flag_status = FLAG_STATUS;                 \
    static constexpr const char* constrains = CONSTRAINS;                   \
    static constexpr bool (*check_func)(const std::string&, std::string&) = \
        nullptr;                                                            \
  }

#define ENV_STRING_STRUCT_STATIC_DEFINITION(NAME) \
  bool NAME::is_cached{false};                    \
  bool NAME::is_defined{false};                   \
  std::string NAME::actual_value{};

#define ENV_STRUCT_DEFINITION_4(NAME, TYPE, DEFAULT_VAL, FLAG_STATUS) \
  ENV_STRUCT_DEFINITION_FULL(                                         \
      NAME, TYPE, DEFAULT_VAL, FLAG_STATUS, nullptr, nullptr)

#define ENV_STRUCT_DEFINITION_6(                                       \
    NAME, TYPE, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CONSTRAINS) \
  ENV_STRUCT_DEFINITION_FULL(                                          \
      NAME, TYPE, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CONSTRAINS)

#define ENV_STRUCT_DEFINITION_WITH_CUSTOM(                                  \
    NAME, TYPE, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CHECK_FUNC)      \
  struct NAME : public std::numeric_limits<TYPE> {                          \
    static bool is_cached;                                                  \
    static bool is_defined;                                                 \
    static TYPE actual_value;                                               \
    static constexpr const char* flag_type = #TYPE;                         \
    static constexpr TYPE default_value = DEFAULT_VAL;                      \
    static constexpr const char* constrains_type = CONSTRAINSTYPE;          \
    static constexpr const char* flag_status = FLAGSTATUS;                  \
    static constexpr const char* constrains = nullptr;                      \
    static constexpr bool (*check_func)(const std::string&, std::string&) = \
        CHECK_FUNC;                                                         \
  }

// Struct defination for non-string env variables with numeric limits
#define ENV_STRUCT_DEFINITION_FULL(                                         \
    NAME, TYPE, DEFAULT_VAL, FLAG_STATUS, CONSTRAINS_TYPE, CONSTRAINS)      \
  struct NAME : public std::numeric_limits<TYPE> {                          \
    static bool is_cached;                                                  \
    static bool is_defined;                                                 \
    static TYPE actual_value;                                               \
    static constexpr const char* flag_type = #TYPE;                         \
    static constexpr TYPE default_value = DEFAULT_VAL;                      \
    static constexpr const char* constrains_type = CONSTRAINS_TYPE;         \
    static constexpr const char* flag_status = FLAG_STATUS;                 \
    static constexpr const char* constrains = CONSTRAINS;                   \
    static constexpr bool (*check_func)(const std::string&, std::string&) = \
        nullptr;                                                            \
  }

#define ENV_STRUCT_STATIC_DEFINITION(NAME, TYPE) \
  bool NAME::is_cached{false};                   \
  bool NAME::is_defined{false};                  \
  TYPE NAME::actual_value{};

template <typename T>
std::string to_string_flexible(const T& value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

bool check_recipe_cache_config(
    const std::string& config,
    std::string& error_msg);

/**
 * @brief Validates a constraint against a specified type and value.
 *
 * This function checks whether the given value satisfies the constraints
 * defined for a specific type and name. It is useful for ensuring that
 * input values conform to expected rules or formats.
 *
 * @param name The name of the constraint to validate.
 * @param type The type associated with the constraint.
 * @param value The value to be validated against the constraint.
 * @param constrains A string representing the constraints to validate against.
 * @return true if the value satisfies the constraints; false otherwise.
 */
template <typename T>
void validate_constraint_with_type(
    const char* name,
    const char* type,
    const T& value,
    const char* constrains);

/**
 * Validates a constraint based on a custom validation function.
 *
 * @param name The name of the constraint to validate.
 * @param value The value of the constraint to validate.
 * @param validator A custom validation function that takes the constraint name
 *                  and value as input and returns a boolean indicating whether
 *                  the validation succeeded. The function may also modify the
 *                  value through its second parameter.
 * @return True if the validation succeeds, false otherwise.
 */
void validate_constraint_custom(
    const char* name,
    const std::string& value,
    std::function<bool(const std::string&, std::string&)> validator);

void check_flag_status(const char* name, const char* flag_status);

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
    enable_if_t<!has_min_max_methods<E>::value, decltype(E::default_value)>
    getenv_E_new(const char* name, const bool& skip_cache) {
  check_flag_status(name, E::flag_status);
  auto result = getenv_by_type_new(
      name,
      skip_cache,
      E::is_cached,
      E::is_defined,
      E::actual_value,
      E::default_value);
  if (result && result[0] != '\0') {
    std::string constrains_type =
        E::constrains_type ? E::constrains_type : CONSTRAINTS_EMPTY;
    if (!constrains_type.empty() && constrains_type == CUSTOM_CONSTRAINT_TYPE) {
      auto value_str = to_string_flexible(result);
      validate_constraint_custom(name, value_str, E::check_func);
    } else {
      validate_constraint_with_type(
          name, E::constrains_type, result, E::constrains);
    }
  }
  return result;
}

template <class E>
typename std::
    enable_if_t<has_min_max_methods<E>::value, decltype(E::default_value)>
    getenv_E_new(const char* name, const bool& skip_cache) {
  check_flag_status(name, E::flag_status);
  auto result = getenv_by_type_new(
      name,
      skip_cache,
      E::is_cached,
      E::is_defined,
      E::actual_value,
      E::default_value,
      E::min(),
      E::max());

  validate_constraint_with_type(
      name, E::constrains_type, result, E::constrains);
  return result;
}

template <class E>
void setenv_E_new(
    const char* name,
    const decltype(E::default_value) new_val,
    int overwrite);

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
