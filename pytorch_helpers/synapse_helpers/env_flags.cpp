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
#include <ostream>
#include <string>
#include <utility>

#include <absl/strings/match.h>

#include "habana_helpers/logging.h"

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
  // 3 | XXX=123         |   123             |
  // 4 | XXX=1234asdf    |   1234            | syntax error "asdf"
  // 5 | XXX=asdf        |   0 or min_val    | syntax error "asdf"
  // 6 | XXX=123...789   |   max_val         | overflow error
  const char* e = getenv(name);
  if (e && *e) {
    errno = 0;
    char* err;
    T env = static_cast<T>(strtonum(e, &err, 0));
    std::string str = std::string(e);
    if (!env) {
      e = (std::string("0x") + str)
              .c_str(); // add 0x prefix to FFFF and such strings to make it
                        // valid which is otherwise invalid.
      env = static_cast<T>(strtonum(
          e,
          &err,
          0)); // converts such valid strings (such as 0xFFFF) to unsinged long.
      if (*err)
        Logger::habana_assert(
            __func__,
            __FILE__,
            static_cast<uint32_t>(__LINE__),
            "Invalid string");
    }
    if (errno) {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          e,
          "\" converted to different value \"",
          env,
          "\" due to overflow.");
    }
    if (*err) {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          e,
          "\" converted to different value \"",
          env,
          "\" due to syntax error \"",
          err,
          '\"');
    }
    if ((env < min_val) || (env > max_val)) {
      auto env_old = env;
      if (env < min_val) {
        env = min_val;
      } else {
        env = max_val;
      }
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          e,
          "\" decoded as ",
          env_old,
          " is out of range <",
          min_val,
          ", ",
          max_val,
          "> and was converted to different value \"",
          env,
          '\"');
    }
    // Return partial conversion result
    // - max_val/min_val in case of overflow/underflow
    // - "123" in "123asdf" case
    return env_value(env);
  } else {
    // Both undefined and XXX= cases
    return default_value(def_val);
  }
}

template <>
RT<bool> getenv_by_type(const char* name, bool def_val) {
  const char* e = getenv(name);
  if (e && *e) {
    bool true_found =
        absl::EqualsIgnoreCase(e, "1") || absl::EqualsIgnoreCase(e, "true");
    bool false_found =
        absl::EqualsIgnoreCase(e, "0") || absl::EqualsIgnoreCase(e, "false");

    if (true_found)
      return env_value(true);
    else if (false_found)
      return env_value(false);
    else {
      PT_SYNHELPER_FATAL(
          "Environment variable \"",
          name,
          "\"=\"",
          e,
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

template <class T, class F>
static T getenv_numeric_new(
    const char* name,
    bool& is_cached,
    bool& is_defined,
    T& act_val,
    T def_val,
    T min_val,
    T max_val,
    F func_strtonum) {
  T env{};
  if (!is_cached) {
    const char* e = getenv(name);
    if (e && *e) {
      errno = 0;
      char* err;
      env = static_cast<T>(func_strtonum(e, &err, 0));
      if (errno) {
        PT_SYNHELPER_FATAL(
            "Environment variable \"",
            name,
            "\"=\"",
            e,
            "\" converted to different value \"",
            env,
            "\" due to overflow.");
      }
      if (*err) {
        PT_SYNHELPER_FATAL(
            "Environment variable \"",
            name,
            "\"=\"",
            e,
            "\" converted to different value \"",
            env,
            "\" due to syntax error \"",
            err,
            '\"');
      }
      if ((env < min_val) || (env > max_val)) {
        auto env_old = env;
        if (env < min_val) {
          env = min_val;
        } else {
          env = max_val;
        }
        PT_SYNHELPER_WARN(
            "Environment variable \"",
            name,
            "\"=\"",
            e,
            "\" decoded as ",
            env_old,
            " is out of range <",
            min_val,
            ", ",
            max_val,
            "> and was converted to different value \"",
            env,
            '\"');
      }
      act_val = env;
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
      bool& is_cached,                          \
      bool& is_defined,                         \
      T& act_val,                               \
      T def_val,                                \
      T min_val,                                \
      T max_val) {                              \
    return getenv_numeric_new(                  \
        name,                                   \
        is_cached,                              \
        is_defined,                             \
        act_val,                                \
        def_val,                                \
        min_val,                                \
        max_val,                                \
        conv);                                  \
  }

INSTANTIATE_GETENV_BY_TYPE_NEW(bool, strtol)
INSTANTIATE_GETENV_BY_TYPE_NEW(int, strtol)
INSTANTIATE_GETENV_BY_TYPE_NEW(long, strtol)
INSTANTIATE_GETENV_BY_TYPE_NEW(unsigned, strtoul)
INSTANTIATE_GETENV_BY_TYPE_NEW(unsigned long, strtoul)
INSTANTIATE_GETENV_BY_TYPE_NEW(long long, strtoll)
INSTANTIATE_GETENV_BY_TYPE_NEW(unsigned long long, strtoull)

ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_MODE, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_LOWERING, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ZERO_STRIDE_SYNTENSOR, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_USE_SYN_TENSOR_IDS, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PRINT_STATS, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PRINT_STATS_DUMP_FREQ, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_PRINT_STATS_TABLE, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_INTERNAL_OLD_SYNAPI, bool);
ENV_STRUCT_STATIC_DEFINITION(HABANA_USE_PERSISTENT_TENSOR, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_LAZY_EAGER_OPTIM_CACHE, unsigned);
ENV_STRUCT_STATIC_DEFINITION(PT_RECIPE_CACHE_IGNORE_VERSION, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_RECIPE_CACHE_DUMP_DEBUG, bool);
ENV_STRUCT_STATIC_DEFINITION(PT_HPU_ENABLE_BUCKET_REFINEMENT, bool);

} // namespace new_style

// ****************************************************************************

} // namespace env_flags
