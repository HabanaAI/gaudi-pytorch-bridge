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

#include "synapse_helpers/env_flags.h"

#include <cerrno>
#include <cstdlib>

#include <algorithm>
#include <cctype>
#include <ostream>
#include <string>
#include <utility>

#include <absl/strings/match.h>

#include "habana_helpers/logging.h"

// NOTE: During PT logger object instantiation the env variables
// like, MOD MASK and TYPE MASK are read using GET_ENV_FLAG macros.
// The PT_MOD* macros except _FATAL are undefined as the use of
// PT logger based macro within this function may cause an infinite loop
// as PT logger object not created yet.
#undef PT_MOD_WARN
#undef PT_MOD_WARN_WITHOUT_LINE_FILE
#undef PT_MOD_BEGIN
#undef PT_MOD_END
#undef PT_MOD_TRACE
#undef PT_MOD_DEBUG

#define PT_MOD_WARN(...) Logger::nop(__VA_ARGS__);
#define PT_MOD_WARN_WITHOUT_LINE_FILE(...) Logger::nop(__VA_ARGS__);
#define PT_MOD_BEGIN(MOD) Logger::nop(MOD);
#define PT_MOD_END(MOD) Logger::nop(MOD);
#define PT_MOD_TRACE(MOD, PNAME, NAME) Logger::nop(MOD, PNAME, NAME);
#define PT_MOD_DEBUG(...) Logger::nop(__VA_ARGS__);

namespace env_flags {

template <class T>
static RT<T> env_value(T v) {
  return std::make_pair(v, true);
}

template <class T>
static RT<T> default_value(T v) {
  return std::make_pair(v, false);
}

// getenv("XXX") returns:
// 1. No XXX definition --> nullptr
// 2. XXX= --> ""
// 3. XXX=asdf --> "asdf"

template <>
RT<const char*> getenv_by_type(const char* name, const char* def_val) {
  // Conversion to string:
  //   |    env var      |   returned value
  // ----------------------------------------
  // 1 | XXX undefined   |   default value
  // 2 | XXX=            |   default value
  // 3 | XXX=asdf        |   "asdf"
  const char* e = getenv(name);
  return e && *e ? env_value(e) : default_value(def_val);
}

template <class T, class F>
static RT<T> getenv_numeric(
    const char* name,
    T def_val,
    T min_val,
    T max_val,
    F strtonum) {
  // Conversion to number:
  //   |    env var      |   returned value  |    LOG
  // -------------------------------------------------------------
  // 1 | XXX undefined   |   default value   |
  // 2 | XXX=            |   default value   |
  // 3 | XXX=123         |   0x7b            |
  // 4 | XXX=abcd        |   0xabcd          |
  // 5 | XXX=0xabcd      |   0xabcd          |
  // 6 | XXX=1234asdf    |   Invalid         | syntax error "asdf"
  // 7 | XXX=asdf        |   Invalid         | syntax error "asdf"
  // 8 | XXX=123...789   |   Invalid         | overflow error
  const char* envstrp = getenv(name);
  if (envstrp && *envstrp) {
    // getenv returned a valid string

    // Only case that we need to handle is hex numbers without 0x prefix
    std::string envstr_lc{envstrp};
    std::string envstr_orig{envstrp};

    // Using a lowercase representation
    std::transform(
        envstr_lc.begin(),
        envstr_lc.end(),
        envstr_lc.begin(),
        [](unsigned char c) { return std::tolower(c); });

    const std::string hex_qual{"0x"};
    if (envstr_lc.find(hex_qual) != 0 &&
        std::any_of(
            std::begin(envstr_lc), std::end(envstr_lc), [](unsigned char c) {
              return (c >= 'a' && c <= 'f');
            })) {
      envstr_lc.insert(0, hex_qual);
      envstrp = envstr_lc.c_str();
    }

    errno = 0;
    char* endptr;
    T envval = static_cast<T>(strtonum(envstrp, &endptr, 0));
    if (errno == ERANGE) {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          envstr_orig,
          "\" converted to different value \"",
          envval,
          "\" due to underflow/overflow.");
    } else if (errno != 0) {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          envstr_orig,
          "\" is not converted properly.");
    }

    // Nonnull endptr means incorrect input string
    // Report syntax error and assert
    if (*endptr) {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          envstr_orig,
          "\" converted to different value \"",
          envval,
          "\" due to syntax error.");
    }

    // Range check and report the error and assert for overflow / underflow
    if ((envval < min_val) || (envval > max_val)) {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          envstr_orig,
          "\" decoded as ",
          envval,
          " is out of range <",
          min_val,
          ", ",
          max_val,
          ">");
    }

    // Return conversion result
    PT_SYNHELPER_DEBUG(
        "Environment variable \"",
        name,
        "\"=\"",
        envstr_orig,
        "\" is decoded as ",
        envval,
        '\"');
    return env_value(envval);
  } else {
    // Both undefined and XXX= cases
    return default_value(def_val);
  }
}

template <>
RT<bool> getenv_by_type(const char* name, bool def_val) {
  const char* envstrp = getenv(name);
  if (envstrp && *envstrp) {
    bool true_found = absl::EqualsIgnoreCase(envstrp, "1") ||
        absl::EqualsIgnoreCase(envstrp, "true");
    bool false_found = absl::EqualsIgnoreCase(envstrp, "0") ||
        absl::EqualsIgnoreCase(envstrp, "false");

    if (true_found)
      return env_value(true);
    else if (false_found)
      return env_value(false);
    else {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          envstrp,
          "\" converted to default value \"",
          def_val,
          "\" due to syntax error");
    }
  }

  return default_value(def_val);
}

#define INST_GETENV_BY_TYPE(T, conv)                                        \
  template <>                                                               \
  RT<T> getenv_by_type(const char* name, T def_val, T min_val, T max_val) { \
    return getenv_numeric(name, def_val, min_val, max_val, conv);           \
  }

INST_GETENV_BY_TYPE(int, strtol)
INST_GETENV_BY_TYPE(long, strtol)
INST_GETENV_BY_TYPE(unsigned, strtoul)
INST_GETENV_BY_TYPE(unsigned long, strtoul)
INST_GETENV_BY_TYPE(long long, strtoll)
INST_GETENV_BY_TYPE(unsigned long long, strtoull)

// ****************************************************************************
// New style of env var declaration

namespace new_style {

const char* getenv_by_type_new(
    const char* name,
    const bool& skip_cache,
    bool& is_cached,
    bool& is_defined,
    std::string& act_val,
    const char* def_val) {
  // Conversion to string:
  //   |    env var      |   returned value
  // ----------------------------------------
  // 1 | XXX undefined   |   default value
  // 2 | XXX=asdf        |   "asdf"
  if (!is_cached || skip_cache) {
    const char* envstrp = getenv(name);
    if (envstrp && *envstrp) {
      act_val = envstrp;
      is_defined = true;
    } else {
      act_val = def_val;
    }
    is_cached = true;
  }
  return act_val.c_str();
}

bool getenv_by_type_new(
    const char* name,
    const bool& skip_cache,
    bool& is_cached,
    bool& is_defined,
    bool& act_val,
    bool def_val,
    bool min_val,
    bool max_val) {
  (void)min_val;
  (void)max_val;
  if (!is_cached || skip_cache) {
    bool result = def_val;
    const char* envstrp = getenv(name);
    if (envstrp && *envstrp) {
      bool true_found = absl::EqualsIgnoreCase(envstrp, "1") ||
          absl::EqualsIgnoreCase(envstrp, "true");
      bool false_found = absl::EqualsIgnoreCase(envstrp, "0") ||
          absl::EqualsIgnoreCase(envstrp, "false");

      if (true_found)
        result = true;
      else if (false_found)
        result = false;
      else {
        PT_SYNHELPER_FATAL(
            "Environment variable \"",
            name,
            "\"=\"",
            envstrp,
            "\" converted to default value \"",
            def_val,
            "\" due to syntax error");
      }
      is_defined = true;
    }
    act_val = result;
    is_cached = true;
  }
  return act_val;
}

template <class T, class F>
static T getenv_numeric_new(
    const char* name,
    const bool& skip_cache,
    bool& is_cached,
    bool& is_defined,
    T& act_val,
    T def_val,
    T min_val,
    T max_val,
    F func_strtonum) {
  // Conversion to number:
  //   |    env var      |   returned value  |    LOG
  // -------------------------------------------------------------
  // 1 | XXX undefined   |   default value   |
  // 2 | XXX=            |   default value   |
  // 3 | XXX=123         |   0x7b            |
  // 4 | XXX=abcd        |   0xabcd          |
  // 5 | XXX=0xabcd      |   0xabcd          |
  // 6 | XXX=1234asdf    |   Invalid         | syntax error "asdf"
  // 7 | XXX=asdf        |   Invalid         | syntax error "asdf"
  // 8 | XXX=123...789   |   Invalid         | overflow error
  T envval{};
  if (!is_cached || skip_cache) {
    const char* envstrp = getenv(name);
    if (envstrp && *envstrp) {
      // getenv returned a valid string

      // Only case that we need to handle is hex numbers without 0x prefix
      std::string envstr_lc{envstrp};
      std::string envstr_orig{envstrp};

      // Using a lowercase representation
      std::transform(
          envstr_lc.begin(),
          envstr_lc.end(),
          envstr_lc.begin(),
          [](unsigned char c) { return std::tolower(c); });

      const std::string hex_qual{"0x"};
      if (envstr_lc.find(hex_qual) != 0 &&
          std::any_of(
              std::begin(envstr_lc), std::end(envstr_lc), [](unsigned char c) {
                return (c >= 'a' && c <= 'f');
              })) {
        envstr_lc.insert(0, hex_qual);
        envstrp = envstr_lc.c_str();
      }

      errno = 0;
      char* endptr;
      envval = static_cast<T>(func_strtonum(envstrp, &endptr, 0));
      if (errno == ERANGE) {
        PT_SYNHELPER_FATAL(
            "Environment variable \"",
            name,
            "\"=\"",
            envstr_orig,
            "\" converted to different value \"",
            envval,
            "\" due to underflow/overflow.");
      } else if (errno != 0) {
        PT_SYNHELPER_FATAL(
            "Environment variable \"",
            name,
            "\"=\"",
            envstr_orig,
            "\" is not converted properly.");
      }

      // Nonnull endptr means incorrect input string
      // Report syntax error and assert
      if (*endptr) {
        PT_SYNHELPER_FATAL(
            "Environment variable \"",
            name,
            "\"=\"",
            envstr_orig,
            "\" converted to different value \"",
            envval,
            "\" due to syntax error.");
      }

      // Range check and report the error and assert for overflow / underflow
      if ((envval < min_val) || (envval > max_val)) {
        PT_SYNHELPER_FATAL(
            "Environment variable \"",
            name,
            "\"=\"",
            envstr_orig,
            "\" decoded as ",
            envval,
            " is out of range <",
            min_val,
            ", ",
            max_val,
            ">");
      }

      // Return conversion result
      PT_SYNHELPER_DEBUG(
          "Environment variable \"",
          name,
          "\"=\"",
          envstr_orig,
          "\" is decoded as ",
          envval,
          '\"');

      act_val = envval;
      is_defined = true;
    } else {
      act_val = def_val;
    }
    is_cached = true;
  }
  return act_val;
}

#define INSTANTIATE_GETENV_BY_TYPE_NEW(T, conv) \
  template <>                                   \
  T getenv_by_type_new(                         \
      const char* name,                         \
      const bool& skip_cache,                   \
      bool& is_cached,                          \
      bool& is_defined,                         \
      T& act_val,                               \
      T def_val,                                \
      T min_val,                                \
      T max_val) {                              \
    return getenv_numeric_new(                  \
        name,                                   \
        skip_cache,                             \
        is_cached,                              \
        is_defined,                             \
        act_val,                                \
        def_val,                                \
        min_val,                                \
        max_val,                                \
        conv);                                  \
  }

INSTANTIATE_GETENV_BY_TYPE_NEW(int, strtol)
INSTANTIATE_GETENV_BY_TYPE_NEW(long, strtol)
INSTANTIATE_GETENV_BY_TYPE_NEW(unsigned, strtoul)
INSTANTIATE_GETENV_BY_TYPE_NEW(unsigned long, strtoul)
INSTANTIATE_GETENV_BY_TYPE_NEW(long long, strtoll)
INSTANTIATE_GETENV_BY_TYPE_NEW(unsigned long long, strtoull)

ENV_STRING_STRUCT_STATIC_DEFINITION(GC_KERNEL_PATH);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_HABANA_MEM_LOG_FILENAME);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_HPU_GRAPH_DUMP_PREFIX);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_RECIPE_CACHE_PATH);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_COMPILATION_STATS_PATH);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_RECIPE_TRACE_PATH);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_HPU_RERUN_JSON_FILE);

ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_MODE, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_ACC_PAR_MODE, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_LOWERING, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DETERMINISTIC_ENABLE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ZERO_STRIDE_SYNTENSOR, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_USE_SYN_TENSOR_IDS, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_MEM_STATS_DUMP, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PRINT_STATS, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PRINT_STATS_DUMP_FREQ, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PRINT_STATS_TABLE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_INTERNAL_OLD_SYNAPI, bool);
ENV_STRUCT_STATIC_DEFINITION(HABANA_USE_PERSISTENT_TENSOR, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_EAGER_OPTIM_CACHE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_RECIPE_CACHE_IGNORE_VERSION, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_RECIPE_CACHE_DUMP_DEBUG, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_COMPILE_THREAD, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_EXECUTION_THREAD, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_INFERENCE_MODE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_LAUNCHTHREAD_USE_THREADPOOL, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_INTER_HOST_CACHING, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_INFERENCE_MODE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_HABANA_CACHING, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_HABANA_STREAMASYNC, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_HOST_MEMORY_CACHE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_HCL_SAME_ADDRESS_RESOLUTION, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_HCL_STREAM, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HABANA_MAX_DMA_COPY_RETRY_COUNT, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HABANA_DMA_COPY_RETRY_DELAY, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_MAX_RECIPE_SUBMISSION_LIMIT, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_HCCL_SLICE_SIZE_MB, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_CACHE_FOLDER_SIZE_MB, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_CACHE_FOLDER_DELETE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_INITIAL_WORKSPACE_SIZE, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_HABANA_POOL_SIZE, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_POOL_STRATEGY, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_POOL_MAX_MERGE_COUNT, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_POOL_ENABLE_LFU_MERGE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HABANA_MEM_LOG_LEVEL, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HABANA_MAX_RECIPE_HIT_COUNT, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_POOL_LOG_FRAGMENTATION_INFO, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_POOL_MEM_ENABLE_TENSOR_INFO, bool)
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_SYNC_OUTPUT_HOST, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_USE_PT_STORE_SYNC, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_USE_NW_STREAM_SYNC, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_EMULATE_DISTRIBUTED, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ERROR_HANDLER, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DONT_USE_STRIDED_VIEW, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_SLICE_INSERT, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_USE_STRIDED_VIEW_FOR_SLICE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_TRANSPOSE_WITH_STRIDED_VIEW, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_PERMUTE_WITH_STRIDED_VIEW, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PRINT_BACKTRACE_ON_SIGNAL, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DUMP_IR_DOT_GRAPH, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DISABLE_INSTANCE_NORM, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DISABLE_ASYNC_COLLECTIVE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_AVOID_RE_EXECUTE_GRAPHS, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_USE_MARKSTEP, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_GRAPH_DUMP, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_SYNLAUNCH_TIME_CAPTURE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_DEBUG_NAMES, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LOWER_AS_STRIDED, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_MAX_ACCUM_SIZE, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_MAX_COMPOUND_OP_SIZE, signed long);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_MAX_COMPOUND_OP_SIZE_SS, signed long);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_STAGE_SUBMISSION, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PGM_ENABLE_CACHE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LOG_MOD_MASK, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LOG_TYPE_MASK, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LOG_NODE_MASK, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_MEMORY_DEFRAGMENTATION, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_DEFRAGMENTATION_INFO, bool);
ENV_STRUCT_STATIC_DEFINITION(
    PT_HPU_MEMORY_DEFRAGMENTATION_RETRIES_LIMIT,
    unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_POOL_MEM_ALLOC_RETRY_WAIT_MS, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_WORKSPACE_MEMORY_SHRINK, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_EAGER_SYN_API, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_SCALAR_MAP_MAXSIZE, unsigned long);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_LAZY_COLLECTIVES, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_SBS, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_FORCE_INDEX_PUT_FRONTEND_FALLBACK, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_MEDIA_PIPE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_H2D_COPY_ASYNC_THREAD, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_H2D_COPY_MIN_TENSOR_SIZE, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_SCALAR_H2D_COPY_MULTIPLE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_SYNAPSE_OUTPUT_PERMUTE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_WEIGHT_CPU_PERMUTE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_MAX_PERMUTE_THRESHOLD, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_FCD_STRIDE_OPT, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_REDUCTION_FLATTEN_INPUT, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HCCL_MEMORY_ALLOWANCE_MB, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_VALID_DATA_RANGE_CHECK, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_NONZERO_CGUID, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_VISUALIZE_GRAPH_INDEX, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_FORCE_USE_DEFAULT_STREAM, bool);
ENV_STRUCT_STATIC_DEFINITION(TRACE_POINT_ENABLE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_GRADIENT_BUCKET_VIEW, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_ENABLE_FP8_CAST_STOCHASTIC_ROUNDING, bool);

// Dynamic shape related env variables
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_DYNAMIC_PASS_FALLBACK, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_DYNAMIC_LAUNCH_FALLBACK, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_MIN_MAX_AS_CURRENT, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DEV_ENABLE_ARANGE_HOST_TENSOR, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DEV_ENABLE_RANDPERM_HOST_TENSOR, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_DEV_ENABLE_PAD_HOST_TENSOR, bool);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_HPU_DYNAMIC_MIN_POLICY_ORDER);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_HPU_DYNAMIC_MAX_POLICY_ORDER);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_VALIDATE_COMPUTE_SHAPE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_ZERO_MIN, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_DISK_CACHE_FOR_DSD, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_FAST_SHAPE_INFERENCE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_RUN_HYBRID_SIF, bool);
ENV_STRING_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLED_JIT_IR_OPS_LIST_FILE);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_UNIQUE_GRAPH, bool);
} // namespace new_style

// ****************************************************************************

} // namespace env_flags
