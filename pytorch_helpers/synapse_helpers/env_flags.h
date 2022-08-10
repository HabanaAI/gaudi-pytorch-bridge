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
  (env_flags::new_style::is_defined_new<env_flags::new_style::e>())

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

ENV_STRING_STRUCT_DEFINITION(GC_KERNEL_PATH, "");
ENV_STRING_STRUCT_DEFINITION(
    PT_HABANA_MEM_LOG_FILENAME,
    "habana_log.livealloc.log");
ENV_STRING_STRUCT_DEFINITION(PT_HPU_GRAPH_DUMP_PREFIX, ".");
// Env var 'PT_RECIPE_CACHE_PATH' to save compiled recipes to disk.
// If proper path is set, disk cache is enabled for all compiled recipes.
ENV_STRING_STRUCT_DEFINITION(PT_RECIPE_CACHE_PATH, "");
ENV_STRING_STRUCT_DEFINITION(PT_COMPILATION_STATS_PATH, "");
ENV_STRING_STRUCT_DEFINITION(PT_HPU_RERUN_JSON_FILE, "");

ENV_STRUCT_DEFINITION(PT_HPU_LAZY_MODE, unsigned, 1);
ENV_STRUCT_DEFINITION(PT_HPU_LAZY_LOWERING, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_DETERMINISTIC_ENABLE, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ZERO_STRIDE_SYNTENSOR, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_USE_SYN_TENSOR_IDS, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_MEM_STATS_DUMP, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_PRINT_STATS, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_PRINT_STATS_DUMP_FREQ, unsigned, 0);
ENV_STRUCT_DEFINITION(PT_HPU_PRINT_STATS_TABLE, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_INTERNAL_OLD_SYNAPI, bool, false);
ENV_STRUCT_DEFINITION(HABANA_USE_PERSISTENT_TENSOR, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_LAZY_EAGER_OPTIM_CACHE, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_COMPILE_THREAD, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_EXECUTION_THREAD, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_LAUNCHTHREAD_USE_THREADPOOL, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT, bool, true);
ENV_STRUCT_DEFINITION(PT_ENABLE_INTER_HOST_CACHING, bool, false);
ENV_STRUCT_DEFINITION(PT_ENABLE_INFERENCE_MODE, bool, true);
ENV_STRUCT_DEFINITION(PT_ENABLE_HABANA_CACHING, bool, true);
ENV_STRUCT_DEFINITION(PT_ENABLE_HABANA_STREAMASYNC, bool, true);
ENV_STRUCT_DEFINITION(PT_ENABLE_HOST_MEMORY_CACHE, bool, true);
ENV_STRUCT_DEFINITION(PT_ENABLE_HCL_SAME_ADDRESS_RESOLUTION, bool, false);
ENV_STRUCT_DEFINITION(PT_ENABLE_HCL_STREAM, bool, true);
ENV_STRUCT_DEFINITION(PT_HABANA_MAX_DMA_COPY_RETRY_COUNT, unsigned, 1000);
ENV_STRUCT_DEFINITION(PT_HABANA_DMA_COPY_RETRY_DELAY, unsigned, 10);
ENV_STRUCT_DEFINITION(PT_HPU_MAX_RECIPE_SUBMISSION_LIMIT, unsigned long, 0);
ENV_STRUCT_DEFINITION(
    PT_HCCL_SLICE_SIZE_MB,
    unsigned,
    DEFAULT_HCCL_SLICE_SIZE_MB);
ENV_STRUCT_DEFINITION(PT_CACHE_FOLDER_SIZE_MB, unsigned, 1024);
ENV_STRUCT_DEFINITION(PT_HABANA_MAX_RECIPE_HIT_COUNT, unsigned, 0);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_SYNC_OUTPUT_HOST, bool, true);
// enable PT_STORE_SYNC if cs-timeouts are seen to perform host synchronization
// before collectives
ENV_STRUCT_DEFINITION(PT_HPU_USE_PT_STORE_SYNC, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_USE_NW_STREAM_SYNC, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_EMULATE_DISTRIBUTED, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ERROR_HANDLER, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_DONT_USE_STRIDED_VIEW, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_USE_STRIDED_VIEW_FOR_SLICE, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_TRANSPOSE_WITH_STRIDED_VIEW, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_PERMUTE_WITH_STRIDED_VIEW, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_PRINT_BACKTRACE_ON_SIGNAL, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_DUMP_IR_DOT_GRAPH, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_DISABLE_INSTANCE_NORM, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_DISABLE_ASYNC_COLLECTIVE, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_AVOID_RE_EXECUTE_GRAPHS, bool, true);
ENV_STRUCT_DEFINITION(PT_USE_MARKSTEP, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_GRAPH_DUMP, unsigned, 0);
ENV_STRUCT_DEFINITION(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_DEBUG_NAMES, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_LOWER_AS_STRIDED, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_MAX_ACCUM_SIZE, unsigned long, UINT64_MAX);
ENV_STRUCT_DEFINITION(PT_HPU_MAX_COMPOUND_OP_SIZE, signed long, INT64_MAX);
ENV_STRUCT_DEFINITION(PT_HPU_MAX_COMPOUND_OP_SIZE_SS, signed long, 10);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_STAGE_SUBMISSION, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_PGM_ENABLE_CACHE, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_LOG_MOD_MASK, unsigned long, UINT64_MAX);
ENV_STRUCT_DEFINITION(PT_HPU_LOG_TYPE_MASK, unsigned long, 1);
ENV_STRUCT_DEFINITION(PT_HPU_LOG_NODE_MASK, unsigned long, 0);
ENV_STRUCT_DEFINITION(PT_HPU_LAZY_EAGER_SYN_API, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_SCALAR_MAP_MAXSIZE, unsigned long, 500);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_LAZY_COLLECTIVES, bool, false);
ENV_STRUCT_DEFINITION(PT_SBS, unsigned, 0);
ENV_STRUCT_DEFINITION(PT_HPU_FORCE_INDEX_PUT_FRONTEND_FALLBACK, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_MEDIA_PIPE, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_NONZERO_CGUID, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_VISUALIZE_GRAPH_INDEX, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT, unsigned, 0);

// Dynamic shape related env variables
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_DYNAMIC_PASS_FALLBACK, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_DYNAMIC_LAUNCH_FALLBACK, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_MIN_MAX_AS_CURRENT, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_DEV_ENABLE_RANDPERM_HOST_TENSOR, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_DEV_ENABLE_PAD_HOST_TENSOR, bool, true);
ENV_STRING_STRUCT_DEFINITION(PT_HPU_DYNAMIC_MIN_POLICY_ORDER, "3,1");
ENV_STRING_STRUCT_DEFINITION(PT_HPU_DYNAMIC_MAX_POLICY_ORDER, "2,3,1");
ENV_STRUCT_DEFINITION(PT_HPU_VALIDATE_COMPUTE_SHAPE, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_ZERO_MIN, bool, false);
// Option to enable Fast shape Inference
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE, bool, false);
// Option to enable UniqueGraph feature
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_UNIQUE_GRAPH, bool, false);
// Option to enable and run Hybrid shape Inference
ENV_STRUCT_DEFINITION(PT_HPU_RUN_HYBRID_SIF, bool, false);
// Option to select JIT IR ops for Hybrid Shape Inference
ENV_STRING_STRUCT_DEFINITION(PT_HPU_ENABLED_JIT_IR_OPS_LIST_FILE, "");

// Options to enable/disable std::copy to async thread for non blocking copy
// and minimum tensor size limit for non blocking copy, Default 1 MB.
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_H2D_COPY_ASYNC_THREAD, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_H2D_COPY_MIN_TENSOR_SIZE, unsigned, 1048576);

// Add scalars to list and copy using synMemCopyAsyncMultiple()
ENV_STRUCT_DEFINITION(PT_HPU_SCALAR_H2D_COPY_MULTIPLE, bool, true);

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

// Device memory related flags
//
ENV_STRUCT_DEFINITION(PT_HPU_INITIAL_WORKSPACE_SIZE, unsigned long, 0);
ENV_STRUCT_DEFINITION(PT_HABANA_POOL_SIZE, unsigned long, 24);
ENV_STRUCT_DEFINITION(PT_HPU_POOL_STRATEGY, unsigned, 5);
ENV_STRUCT_DEFINITION(PT_HPU_POOL_MAX_MERGE_COUNT, unsigned, 5);
ENV_STRUCT_DEFINITION(PT_HPU_POOL_ENABLE_LFU_MERGE, bool, true);
ENV_STRUCT_DEFINITION(PT_HABANA_MEM_LOG_LEVEL, unsigned, 0);
ENV_STRUCT_DEFINITION(PT_HPU_POOL_LOG_FRAGMENTATION_INFO, bool, false);
// G1: 4 buffers, 6 communicators, internal slicing at 16 MB (16*4*6=384) per
// stream
ENV_STRUCT_DEFINITION(PT_HCCL_MEMORY_ALLOWANCE_MB, unsigned, 384);
ENV_STRUCT_DEFINITION(PT_ENABLE_MEMORY_DEFRAGMENTATION, bool, false);
ENV_STRUCT_DEFINITION(PT_ENABLE_DEFRAGMENTATION_INFO, bool, false);
ENV_STRUCT_DEFINITION(PT_HPU_MEMORY_DEFRAGMENTATION_RETRIES_LIMIT, unsigned, 5);
ENV_STRUCT_DEFINITION(PT_HPU_POOL_MEM_THRESHOLD_PERC, unsigned, 95);

// Option to pass layouts handling to new flow, which will be handled by Syanpse
// No more need to permute layouts/weight via jit optimization passes.
// All will be handled via Synapse
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE, bool, true);
ENV_STRUCT_DEFINITION(PT_HPU_MAX_PERMUTE_THRESHOLD, unsigned, 5);
ENV_STRUCT_DEFINITION(PT_HPU_FCD_STRIDE_OPT, bool, true);

// Option to enable the flattening algorithm for the reduction ops
ENV_STRUCT_DEFINITION(PT_HPU_REDUCTION_FLATTEN_INPUT, bool, true);

// Option to enable numeric limits check for CPU tensors when downcasting
// Ex long/double data type to int/float data type
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_VALID_DATA_RANGE_CHECK, bool, true);

// MultiUser stream - flag to force all ops to default stream
ENV_STRUCT_DEFINITION(PT_HPU_FORCE_USE_DEFAULT_STREAM, bool, false);
// Host Trace
ENV_STRUCT_DEFINITION(TRACE_POINT_ENABLE, bool, false);

// gradient bucket feature
ENV_STRUCT_DEFINITION(PT_HPU_ENABLE_GRADIENT_BUCKET_VIEW, bool, false);

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
bool is_defined_new() {
  return E::is_defined;
}

} // namespace new_style

// ****************************************************************************

} // namespace env_flags
