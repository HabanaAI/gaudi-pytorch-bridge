/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#define GET_ENV_FLAG_NEW(e) \
  (env_flags::new_style::get_env_flag_new<env_flags::new_style::e>(#e))
#define SET_ENV_FLAG_NEW(e, v, o) \
  (env_flags::new_style::set_env_flag_new<env_flags::new_style::e>(#e, v, o))
#define UNSET_ENV_FLAG_NEW(e) \
  (env_flags::new_style::unset_env_flag_new<env_flags::new_style::e>(#e))
#define IS_ENV_FLAG_DEFINED_NEW(e) \
  (env_flags::new_style::is_defined_new<env_flags::new_style::e>())

// ****************************************************************************

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

struct PT_HCCL_SLICE_SIZE_MB : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 128;
};

struct PT_HPU_INITIAL_WORKSPACE_SIZE
    : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = 0;
};

struct PT_HABANA_MAX_DMA_COPY_RETRY_COUNT
    : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 1000;
};

// Synapse-specific env var.
// Colon-separated list of tpc kernel libs to be loaded for GC
struct GC_KERNEL_PATH {
  static constexpr const char* default_value = "";
};

struct PT_HABANA_DMA_COPY_RETRY_DELAY : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 10;
};

struct PT_HABANA_POOL_SIZE : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = 24;
};

struct PT_HPU_POOL_STRATEGY : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 5;
};

struct PT_HPU_POOL_MAX_MERGE_COUNT : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 5;
};

struct PT_HPU_POOL_ENABLE_LFU_MERGE {
  static constexpr bool default_value = true;
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

struct PT_HPU_POOL_LOG_FRAGMENTATION_INFO {
  static constexpr bool default_value = false;
};

struct PT_ENABLE_SYNC_OUTPUT_HOST {
  static constexpr bool default_value = true;
};

struct PT_HPU_ENABLE_SYNC_OUTPUT_HOST {
  static constexpr bool default_value = true;
};

struct PT_USE_HCL_SYNC {
  static constexpr bool default_value = true;
};

struct PT_HPU_USE_HCL_SYNC {
  static constexpr bool default_value = true;
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

struct PT_HPU_ERROR_HANDLER {
  static constexpr bool default_value = true;
};

struct PT_HPU_PRINT_BACKTRACE_ON_SIGNAL {
  static constexpr bool default_value = true;
};

struct PT_HPU_DUMP_IR_DOT_GRAPH {
  static constexpr bool default_value = false;
};

struct PT_HPU_DISABLE_INSTANCE_NORM {
  static constexpr bool default_value = false;
};

struct PT_HPU_AVOID_RE_EXECUTE_GRAPHS {
  static constexpr bool default_value = true;
};

// Experimental feature for media pipe. Do not document for end customers
struct PT_HPU_ENABLE_DATAPTR_ACCESS {
  static constexpr bool default_value = false;
};

struct PT_HPU_MAX_RECIPE_SUBMISSION_LIMIT
    : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = 0;
};

struct PT_HPU_MAX_ACCUM_SIZE : public std::numeric_limits<std::size_t> {
  static constexpr std::size_t default_value =
      std::numeric_limits<std::size_t>::max();
};

struct PT_HPU_GRAPH_DUMP : public std::numeric_limits<unsigned> {
  static constexpr unsigned default_value = 0;
};

struct PT_HPU_GRAPH_DUMP_PREFIX {
  static constexpr const char* default_value = ".";
};

struct PT_ENABLE_SYNLAUNCH_TIME_CAPTURE {
  static constexpr bool default_value = false;
};

struct PT_HPU_LOG_MOD_MASK : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = UINT64_MAX;
};

struct PT_HPU_LOG_TYPE_MASK : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = 3;
};

struct PT_HPU_PGM_ENABLE_CACHE : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = 0;
};

struct PT_HPU_LOG_NODE_MASK : public std::numeric_limits<unsigned long> {
  static constexpr unsigned long default_value = 0;
};
struct PT_HPU_ENABLE_DEBUG_NAMES {
  static constexpr bool default_value = true;
};

// Option to save compiled recipes to disk.
// If proper path is set, disk cache is enabled for all compiled recipes.
struct PT_RECIPE_CACHE_PATH {
  static constexpr const char* default_value = "";
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

ENV_STRUCT_DEFINITION(PT_HPU_LAZY_MODE, unsigned, 2);
ENV_STRUCT_DEFINITION(PT_HPU_LAZY_LOWERING, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ZERO_STRIDE_SYNTENSOR, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_USE_SYN_TENSOR_IDS, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_PRINT_STATS, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_PRINT_STATS_DUMP_FREQ, unsigned, 0);
ENV_STRUCT_DEFINITION(PT_HPU_PRINT_STATS_TABLE, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_INTERNAL_OLD_SYNAPI, bool, false);
ENV_STRUCT_DEFINITION(HABANA_USE_PERSISTENT_TENSOR, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_LAZY_EAGER_OPTIM_CACHE, unsigned, 1);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_BUCKET_REFINEMENT, bool, false);

// Option to skip cache versioning mechanism.
// This will skip the check of Libs and Env compatibility of serialized recipes
// read from disk.
ENV_STRUCT_DEFINITION(PT_RECIPE_CACHE_IGNORE_VERSION, bool, false);

// Option to dump additional debug information to disk cache directory.
// This works only with PT_RECIPE_CACHE_PATH set.
// In the disk cache folder for every recipe, '<hash>.hash_content' files are
// dumped. These files contain all the information that contribute to hash of a
// given recipe and can be used i.e. in cases when graphs are expected to
// produce exactly the same cache entires.
ENV_STRUCT_DEFINITION(PT_RECIPE_CACHE_DUMP_DEBUG, bool, false);

template <class T>
T getenv_by_type_new(
    const char* name,
    bool& is_cached,
    bool& is_defined,
    T& act_val,
    const T def_val,
    const T min_val,
    const T max_val);

template <class T>
void setenv_by_type_new(
    const char* name,
    bool& is_cached,
    bool& is_defined,
    T& act_val,
    const T new_val,
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
    enable_if<has_min_max_methods<E>::value, decltype(E::default_value)>::type
    getenv_E_new(const char* name) {
  return getenv_by_type_new(
      name,
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
decltype(E::default_value) get_env_flag_new(const char* name) {
  return getenv_E_new<E>(name);
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
bool is_defined_new() {
  return E::is_defined;
}

} // namespace new_style

// ****************************************************************************

} // namespace env_flags
