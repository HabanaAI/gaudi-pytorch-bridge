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

#include <absl/strings/str_format.h>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/profiling/profiling.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/runtime_tracing.h"

#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <hl_logger/hllog.hpp>

#define BRACED_PARAM(p) "{}"
#define FORMAT_AND_MSG(...) \
  HLLOG_APPLY(HLLOG_EMPTY, BRACED_PARAM, ##__VA_ARGS__), ##__VA_ARGS__

namespace HlLogger {
// Define an enum with all the logger types the last item must be LOG_MAX
// If you add any new LoggerType entry, please do it also for ModuleMask
enum class LoggerType {
  PT_DEVICE,
  PT_KERNEL,
  PT_BRIDGE,
  PT_SYNHELPER,
  PT_DISTRIBUTED,
  PT_LAZY,
  PT_TRACE,
  PT_FALLBACK,
  PT_STATS,
  PT_TEST,
  PT_DYNAMIC_SHAPE,
  PT_DEVMEM,
  PT_HABHELPER,
  PT_IRGRAPH,
  PT_VIEWTABLE,
  PT_REFINEMENT,
  PT_HOSTSTAT,
  PT_LAYOUTS,
  PT_PARALLEL_ACC,
  PT_LAZY_EAGER,
  PT_MEMLOG,
  PT_EXEC_THREAD,
  PT_EAGER,
  PT_CUSTOM, // Don't use it in checkin code.
  PT_RECIPE_STATS,
  LOG_MAX // Don't use it
};
} // namespace HlLogger
#define HLLOG_ENUM_TYPE_NAME HlLogger::LoggerType

// Redefining c10 StringUtils functions here as distributed and syn
// helpers are independent of  torch libraries
namespace Logger {

inline bool isTracingForced(
    const HlLogger::LoggerType& mod,
    std::string_view name) {
  static std::unordered_map<HlLogger::LoggerType, uint64_t> mask_map{
      {HlLogger::LoggerType::PT_DEVICE, 0x1},
      {HlLogger::LoggerType::PT_KERNEL, 0x2},
      {HlLogger::LoggerType::PT_BRIDGE, 0x4},
      {HlLogger::LoggerType::PT_SYNHELPER, 0x8},
      {HlLogger::LoggerType::PT_DISTRIBUTED, 0x10},
      {HlLogger::LoggerType::PT_LAZY, 0x20},
      {HlLogger::LoggerType::PT_TRACE, 0x40},
      {HlLogger::LoggerType::PT_FALLBACK, 0x80},
      {HlLogger::LoggerType::PT_STATS, 0x100},
      {HlLogger::LoggerType::PT_TEST, 0x200},
      {HlLogger::LoggerType::PT_DYNAMIC_SHAPE, 0x400},
      {HlLogger::LoggerType::PT_DEVMEM, 0x800},
      {HlLogger::LoggerType::PT_HABHELPER, 0x1000},
      {HlLogger::LoggerType::PT_IRGRAPH, 0x2000},
      {HlLogger::LoggerType::PT_VIEWTABLE, 0x4000},
      {HlLogger::LoggerType::PT_REFINEMENT, 0x8000},
      {HlLogger::LoggerType::PT_HOSTSTAT, 0x10000},
      {HlLogger::LoggerType::PT_LAYOUTS, 0x20000},
      {HlLogger::LoggerType::PT_PARALLEL_ACC, 0x40000},
      {HlLogger::LoggerType::PT_LAZY_EAGER, 0x80000},
      {HlLogger::LoggerType::PT_MEMLOG, 0x100000},
      {HlLogger::LoggerType::PT_EXEC_THREAD, 0x200000},
      {HlLogger::LoggerType::PT_EAGER, 0x400000},
      {HlLogger::LoggerType::PT_CUSTOM,
       0x800000}, // Don't use it in checkin code.
      {HlLogger::LoggerType::PT_RECIPE_STATS, 0x1000000},
      {HlLogger::LoggerType::LOG_MAX, 0x2000000} // Don't use it
  };

  if (GET_ENV_FLAG_NEW(PT_FORCED_TRACING_MASK) & mask_map[mod])
    // forced synapse logger
    return true;
  else
    // tensorboard enabled
    return habana::profile::bridge::is_enabled(name);
}

inline std::string DebugString(const HlLogger::LoggerType& mod) {
  static std::unordered_map<HlLogger::LoggerType, std::string> names = {
      {HlLogger::LoggerType::PT_DEVICE, "PT_DEVICE"},
      {HlLogger::LoggerType::PT_KERNEL, "PT_KERNEL"},
      {HlLogger::LoggerType::PT_BRIDGE, "PT_BRIDGE"},
      {HlLogger::LoggerType::PT_SYNHELPER, "PT_SYNHELPER"},
      {HlLogger::LoggerType::PT_DISTRIBUTED, "PT_DISTRIBUTED"},
      {HlLogger::LoggerType::PT_LAZY, "PT_LAZY"},
      {HlLogger::LoggerType::PT_TRACE, "PT_TRACE"},
      {HlLogger::LoggerType::PT_FALLBACK, "PT_FALLBACK"},
      {HlLogger::LoggerType::PT_STATS, "PT_STATS"},
      {HlLogger::LoggerType::PT_TEST, "PT_TEST"},
      {HlLogger::LoggerType::PT_DYNAMIC_SHAPE, "PT_DYNAMIC_SHAPE"},
      {HlLogger::LoggerType::PT_DEVMEM, "PT_DEVMEM"},
      {HlLogger::LoggerType::PT_HABHELPER, "PT_HABHELPER"},
      {HlLogger::LoggerType::PT_IRGRAPH, "PT_IRGRAPH"},
      {HlLogger::LoggerType::PT_VIEWTABLE, "PT_VIEWTABLE"},
      {HlLogger::LoggerType::PT_REFINEMENT, "PT_REFINEMENT"},
      {HlLogger::LoggerType::PT_HOSTSTAT, "PT_HOSTSTAT"},
      {HlLogger::LoggerType::PT_LAYOUTS, "PT_LAYOUTS"},
      {HlLogger::LoggerType::PT_PARALLEL_ACC, "PT_PARALLEL_ACC"},
      {HlLogger::LoggerType::PT_LAZY_EAGER, "PT_LAZY_EAGER"},
      {HlLogger::LoggerType::PT_MEMLOG, "PT_MEMLOG"},
      {HlLogger::LoggerType::PT_EXEC_THREAD, "PT_EXEC_THREAD"},
      {HlLogger::LoggerType::PT_EAGER, "PT_EAGER"},
      {HlLogger::LoggerType::PT_RECIPE_STATS, "PT_RECIPE_STATS"},
      {HlLogger::LoggerType::PT_CUSTOM, "PT_CUSTOM"},
      // {HlLogger::LoggerType::LOG_MAX, "LOG_MAX"},
  };
  if (auto result = names.find(mod); result != names.end())
    return result->second;
  else
    return std::string("UNDEFINED");
};

template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

template <size_t N>
struct CanonicalizeStrTypes<std::array<char, N>> {
  using type = const char*;
};

inline std::ostream& _str(std::ostream& ss) {
  return ss;
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  ss << t;
  return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

template <typename... Args>
inline std::string _str_wrapper(const Args&... args) {
  std::ostringstream ss;
  _str(ss, args...);
  return ss.str();
}

uint64_t get_tid_internal();

inline uint64_t get_tid() {
  static thread_local uint64_t tid{static_cast<uint64_t>(get_tid_internal())};
  return tid;
}

uint64_t get_rank_internal();

inline uint64_t get_rank() {
  static uint64_t tid{static_cast<uint64_t>(get_rank_internal())};
  return tid;
}

inline void append_hdr(std::string& result) {
  auto timeSinceEpoch = std::chrono::system_clock::now().time_since_epoch();
  auto epochSeconds =
      std::chrono::duration_cast<std::chrono::seconds>(timeSinceEpoch);
  std::chrono::microseconds usecs =
      std::chrono::duration_cast<std::chrono::microseconds>(timeSinceEpoch) -
      epochSeconds;
  time_t unixTimestamp = epochSeconds.count();
  struct tm ltime;
  localtime_r(&unixTimestamp, &ltime);
  absl::StrAppendFormat(
      &result,
      "[%02d-%02d %02d:%02d:%02d::%06d][R%03d][%ld]",
      ltime.tm_mon + 1,
      ltime.tm_mday,
      ltime.tm_hour,
      ltime.tm_min,
      ltime.tm_sec,
      usecs.count(),
      get_rank(),
      get_tid());
}

inline std::string print_hdr() {
  std::string result;
  // We're likely going to append more stuff so reserve space up front.
  //(header alone is longer than SSO)
  result.reserve(256);
  append_hdr(result);
  return result;
}

template <typename... Args>
inline void print(std::ostream& os, const Args&... args) {
  os << Logger::print_hdr();
  (os << ... << args);
}

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline std::string str(const Args&... args) {
  return _str_wrapper<typename CanonicalizeStrTypes<Args>::type...>(args...);
}

// Specializations for already-a-string types.
template <>
inline std::string str(const std::string& str) {
  return str;
}

inline std::string str(const char* c_str) {
  return c_str;
}

// Unpack msg
template <typename... Args>
decltype(auto) CheckMsgImpl(const char*, const Args&... args) {
  return Logger::str(args...);
}

inline const char* CheckMsgImpl(const char* msg) {
  return msg;
}

inline const char* CheckMsgImpl(const char*, const char* args) {
  return args;
}

[[noreturn]] void habana_assert(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg);

template <class... Args>
inline void nop(__attribute__((unused)) const Args&... args){};
} // namespace Logger

class PTFuncLog {
 private:
  const std::string_view module;
  const std::string_view pName;
  const std::string_view name;
  bool isActive;

 public:
  PTFuncLog(
      const std::string_view module,
      const std::string_view pn,
      const std::string_view n,
      bool isActive)
      : module(module), pName(pn), name(n), isActive(isActive) {
    if (isActive) {
      auto message{Logger::print_hdr()};
      HLLOG_TRACE(
          PT_TRACE, FORMAT_AND_MSG(message, module, ": begin of ", pName));
    }
    synapse_helpers::trace_start(name.data());
    habana::profile::bridge::trace_start(name);
  }
  ~PTFuncLog() {
    if (isActive) {
      auto message{Logger::print_hdr()};
      HLLOG_TRACE(
          PT_TRACE, FORMAT_AND_MSG(message, module, ": end of ", pName));
    }
    synapse_helpers::trace_end(name.data());
    habana::profile::bridge::trace_end(name);
  }
};

#define HABANA_CHECK_MSG(cond, ...) \
  Logger::CheckMsgImpl(             \
      "Expected " #cond " to be true, but got false.", ##__VA_ARGS__)

#define HABANA_ASSERT(condition, ...)                                      \
  if (__builtin_expect(static_cast<bool>(!(condition)), 0)) {              \
    auto MSG_ = std::string(HABANA_CHECK_MSG(condition, ##__VA_ARGS__));   \
    HLLOG_ERR_F(PT_BRIDGE, FORMAT_AND_MSG(__FILE__, ":", __LINE__, MSG_)); \
    hl_logger::logStacktrace(                                              \
        HlLogger::LoggerType::PT_BRIDGE, HLLOG_LEVEL_ERROR);               \
    Logger::habana_assert(                                                 \
        __func__, __FILE__, static_cast<uint32_t>(__LINE__), MSG_);        \
  }

/************************CRITICAL MACROS************************/
#define PT_MOD_FATAL(MOD, ...)                                                 \
  {                                                                            \
    HLLOG_CRITICAL_F(                                                          \
        MOD,                                                                   \
        FORMAT_AND_MSG(                                                        \
            __FILE__,                                                          \
            ":",                                                               \
            __LINE__,                                                          \
            "\nrank: ",                                                        \
            Logger::get_rank(),                                                \
            "\ntid: ",                                                         \
            Logger::get_tid(),                                                 \
            __VA_ARGS__));                                                     \
    hl_logger::logStacktrace(HlLogger::LoggerType::MOD, HLLOG_LEVEL_CRITICAL); \
    Logger::habana_assert(                                                     \
        __func__,                                                              \
        __FILE__,                                                              \
        static_cast<uint32_t>(__LINE__),                                       \
        Logger::str(                                                           \
            Logger::print_hdr(),                                               \
            "FATAL ERROR :: MODULE:",                                          \
            Logger::DebugString(HlLogger::LoggerType::MOD),                    \
            " ",                                                               \
            __VA_ARGS__));                                                     \
  }

#define PT_DEVICE_FATAL(...) PT_MOD_FATAL(PT_DEVICE, __VA_ARGS__)
#define PT_KERNEL_FATAL(...) PT_MOD_FATAL(PT_KERNEL, __VA_ARGS__)
#define PT_BRIDGE_FATAL(...) PT_MOD_FATAL(PT_BRIDGE, __VA_ARGS__)
#define PT_SYNHELPER_FATAL(...) PT_MOD_FATAL(PT_SYNHELPER, __VA_ARGS__)
#define PT_HABHELPER_FATAL(...) PT_MOD_FATAL(PT_HABHELPER, __VA_ARGS__)
#define PT_DEVMEM_FATAL(...) PT_MOD_FATAL(PT_DEVMEM, __VA_ARGS__)
#define PT_DISTRIBUTED_FATAL(...) PT_MOD_FATAL(PT_DISTRIBUTED, __VA_ARGS__)
#define PT_LAZY_FATAL(...) PT_MOD_FATAL(PT_LAZY, __VA_ARGS__)
#define PT_IRGRAPH_FATAL(...) PT_MOD_FATAL(PT_IRGRAPH, __VA_ARGS__)
#define PT_DYNAMIC_SHAPE_FATAL(...) PT_MOD_FATAL(PT_DYNAMIC_SHAPE, __VA_ARGS__)
#define PT_LAZY_EAGER_FATAL(...) PT_MOD_FATAL(PT_LAZY_EAGER, __VA_ARGS__)
#define PT_EAGER_FATAL(...) PT_MOD_FATAL(PT_EAGER, __VA_ARGS__)

/************************WARNING MACROS************************/
#define PT_MOD_WARN(MOD, ...) \
  HLLOG_WARN(MOD, FORMAT_AND_MSG(__FILE__, ":", __LINE__, "\t", __VA_ARGS__));
#define PT_MOD_WARN_WITHOUT_LINE_FILE(MOD, ...) \
  HLLOG_WARN(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define PT_BRIDGE_WARN(...) PT_MOD_WARN(PT_BRIDGE, __VA_ARGS__)
#define PT_DEVICE_WARN(...) PT_MOD_WARN(PT_DEVICE, __VA_ARGS__)
#define PT_EAGER_WARN(...) PT_MOD_WARN(PT_EAGER, __VA_ARGS__)
#define PT_KERNEL_WARN(...) PT_MOD_WARN(PT_KERNEL, __VA_ARGS__)
#define PT_SYNHELPER_WARN(...) PT_MOD_WARN(PT_SYNHELPER, __VA_ARGS__)
#define PT_HABHELPER_WARN(...) PT_MOD_WARN(PT_HABHELPER, __VA_ARGS__)
#define PT_DEVMEM_WARN(...) PT_MOD_WARN(PT_DEVMEM, __VA_ARGS__)
#define PT_DISTRIBUTED_WARN(...) PT_MOD_WARN(PT_DISTRIBUTED, __VA_ARGS__)
#define PT_LAZY_WARN(...) PT_MOD_WARN(PT_LAZY, __VA_ARGS__)
#define PT_IRGRAPH_WARN(...) PT_MOD_WARN(PT_IRGRAPH, __VA_ARGS__)
#define PT_FALLBACK_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PT_FALLBACK, __VA_ARGS__)
#define PT_TEST_WARN(...) PT_MOD_WARN_WITHOUT_LINE_FILE(PT_TEST, __VA_ARGS__)
#define PT_DYNAMIC_SHAPE_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PT_DYNAMIC_SHAPE, __VA_ARGS__)
#define PT_LAZY_EAGER_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PT_LAZY_EAGER, __VA_ARGS__)

/************************TRACE MACROS************************************/
#define PT_MOD_BEGIN(MOD) PT_MOD_SCOPE(MOD, __PRETTY_FUNCTION__, __FUNCTION__)

#define PT_DEVICE_BEGIN PT_MOD_BEGIN(PT_DEVICE)
#define PT_KERNEL_BEGIN                                           \
  {                                                               \
    bool lazy_mode = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);          \
    HABANA_ASSERT(                                                \
        !lazy_mode,                                               \
        "Lazy Mode = ",                                           \
        lazy_mode,                                                \
        "  :  "                                                   \
        "Please avoid Legacy eager calls in Lazy execution mode " \
        "(for optimizers use PT_OPTIMIZER_KERNEL_BEGIN),"         \
        " for other kernels use PT_OTHER_KERNEL_BEGIN");          \
    PT_MOD_BEGIN(PT_KERNEL)                                       \
  }
// following macro is a non-asserting version of PT_KERNEL_BEGIN
#define PT_OTHER_OPS_BEGIN PT_MOD_BEGIN(PT_KERNEL)
#define PT_BRIDGE_BEGIN PT_MOD_BEGIN(PT_BRIDGE)
#define PT_SYNHELPER_BEGIN PT_MOD_BEGIN(PT_SYNHELPER)
#define PT_HABHELPER_BEGIN PT_MOD_BEGIN(PT_HABHELPER)
#define PT_DEVMEM_BEGIN PT_MOD_BEGIN(PT_DEVMEM)
#define PT_DISTRIBUTED_BEGIN PT_MOD_BEGIN(PT_DISTRIBUTED)
#define PT_LAZY_BEGIN PT_MOD_BEGIN(PT_LAZY)

#define PT_MOD_END(MOD)

#define PT_DEVICE_END PT_MOD_END(PT_DEVICE)
#define PT_KERNEL_END PT_MOD_END(PT_KERNEL)
#define PT_OTHER_OPS_END PT_MOD_END(PT_KERNEL)
#define PT_BRIDGE_END PT_MOD_END(PT_BRIDGE)
#define PT_SYNHELPER_END PT_MOD_END(PT_SYNHELPER)
#define PT_HABHELPER_END PT_MOD_END(PT_HABHELPER)
#define PT_DEVMEM_END PT_MOD_END(PT_DEVMEM)
#define PT_DISTRIBUTED_END PT_MOD_END(PT_DISTRIBUTED)
#define PT_LAZY_END PT_MOD_END(PT_LAZY)

#define PT_MOD_SCOPE(MOD, PNAME, NAME)                             \
  std::optional<PTFuncLog> ptFuncLogger{};                         \
  bool isTraceLoggerEnabled{hl_logger::logLevelAtLeast(            \
      HlLogger::LoggerType::MOD, HLLOG_LEVEL_TRACE)};              \
  bool isTracingForced{                                            \
      Logger::isTracingForced(HlLogger::LoggerType::MOD, NAME)};   \
  if (isTraceLoggerEnabled or isTracingForced) {                   \
    ptFuncLogger.emplace(#MOD, PNAME, NAME, isTraceLoggerEnabled); \
  }

#define PT_MOD_TRACE(MOD, PNAME, NAME) PT_MOD_SCOPE(MOD, PNAME, NAME)

#define PT_EAGER_TRACE PT_MOD_TRACE(PT_EAGER, __PRETTY_FUNCTION__, __FUNCTION__)

#define PT_LAZY_TRACE PT_MOD_TRACE(PT_LAZY, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_LAZY_TRACE_WITH_NAME(name) PT_MOD_TRACE(PT_LAZY, name, name)
#define PT_BRIDGE_TRACE \
  PT_MOD_TRACE(PT_BRIDGE, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_FALLBACK_TRACE \
  PT_MOD_TRACE(PT_FALLBACK, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_SYNHELPER_TRACE \
  PT_MOD_TRACE(PT_SYNHELPER, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_HABHELPER_TRACE \
  PT_MOD_TRACE(PT_HABHELPER, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_TEST_TRACE PT_MOD_TRACE(PT_TEST, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_DYNAMIC_SHAPE_TRACE \
  PT_MOD_TRACE(PT_DYNAMIC_SHAPE, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_DEVMEM_TRACE \
  PT_MOD_TRACE(PT_DEVMEM, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_LAZY_EAGER_TRACE \
  PT_MOD_TRACE(PT_LAZY_EAGER, __PRETTY_FUNCTION__, __FUNCTION__)

/************************DEBUG MACROS************************************/
#define IS_MOD_DEBUG_ENABLED(MOD) \
  hl_logger::logLevelAtLeast(HlLogger::LoggerType::MOD, HLLOG_LEVEL_DEBUG)
#define IS_MEMLOG_DEBUG_ENABLED IS_MOD_DEBUG_ENABLED(PT_MEMLOG)
#define IS_SYNHELPER_DEBUG_ENABLED IS_MOD_DEBUG_ENABLED(PT_SYNHELPER)
#define IS_BRIDGE_DEBUG_ENABLED IS_MOD_DEBUG_ENABLED(PT_BRIDGE)

#define PT_MOD_DEBUG(MOD, ...) HLLOG_DEBUG(MOD, FORMAT_AND_MSG(__VA_ARGS__));

#define PT_PROFILE_DUMP(...) \
  HLLOG_INFO(PT_RECIPE_STATS, FORMAT_AND_MSG(__VA_ARGS__))

#define PT_BRIDGE_DEBUG(...) PT_MOD_DEBUG(PT_BRIDGE, __VA_ARGS__)
#define PT_CUSTOM_DEBUG(...) PT_MOD_DEBUG(PT_CUSTOM, __VA_ARGS__)
#define PT_DEVICE_DEBUG(...) PT_MOD_DEBUG(PT_DEVICE, __VA_ARGS__)
#define PT_DEVMEM_DEBUG(...) PT_MOD_DEBUG(PT_DEVMEM, __VA_ARGS__)
#define PT_DISTRIBUTED_DEBUG(...) PT_MOD_DEBUG(PT_DISTRIBUTED, __VA_ARGS__)
#define PT_DYNAMIC_SHAPE_DEBUG(...) PT_MOD_DEBUG(PT_DYNAMIC_SHAPE, __VA_ARGS__)
#define PT_EAGER_DEBUG(...) PT_MOD_DEBUG(PT_EAGER, __VA_ARGS__)
#define PT_FALLBACK_DEBUG(...) PT_MOD_DEBUG(PT_FALLBACK, __VA_ARGS__)
#define PT_HABHELPER_DEBUG(...) PT_MOD_DEBUG(PT_HABHELPER, __VA_ARGS__)
#define PT_HOSTSTAT_DEBUG(...) PT_MOD_DEBUG(PT_HOSTSTAT, __VA_ARGS__)
#define PT_IRGRAPH_DEBUG(...) PT_MOD_DEBUG(PT_IRGRAPH, __VA_ARGS__)
#define PT_KERNEL_DEBUG(...) PT_MOD_DEBUG(PT_KERNEL, __VA_ARGS__)
#define PT_LAYOUTS_DEBUG(...) PT_MOD_DEBUG(PT_LAYOUTS, __VA_ARGS__)
#define PT_LAZY_DEBUG(...) PT_MOD_DEBUG(PT_LAZY, __VA_ARGS__)
#define PT_LAZY_EAGER_DEBUG(...) PT_MOD_DEBUG(PT_LAZY_EAGER, __VA_ARGS__)
#define PT_LAZY_EXEC_THREAD(...) PT_MOD_DEBUG(PT_EXEC_THREAD, __VA_ARGS__)
#define PT_LAZY_PARALLEL_ACC_DEBUG(...) \
  PT_MOD_DEBUG(PT_PARALLEL_ACC, __VA_ARGS__)
#define PT_MEMLOG_DEBUG(...) PT_MOD_DEBUG(PT_MEMLOG, __VA_ARGS__)
#define PT_REFINEMENT_DEBUG(...) PT_MOD_DEBUG(PT_REFINEMENT, __VA_ARGS__)
#define PT_SYNHELPER_DEBUG(...) PT_MOD_DEBUG(PT_SYNHELPER, __VA_ARGS__)
#define PT_TEST_DEBUG(...) PT_MOD_DEBUG(PT_TEST, __VA_ARGS__)
#define PT_VIEWTABLE_DEBUG(...) PT_MOD_DEBUG(PT_VIEWTABLE, __VA_ARGS__)

#define PT_TEST_DEBUG_TH(...)     \
  PT_TEST_DEBUG(                  \
      "PTI_DBG :: ",              \
      __FUNCTION__,               \
      ":",                        \
      __LINE__,                   \
      " THR=",                    \
      std::this_thread::get_id(), \
      " :: ",                     \
      __VA_ARGS__)

#define PT_OP_INFO(...) HLLOG_INFO(PT_STATS, FORMAT_AND_MSG(__VA_ARGS__));
#define PT_FALLBACK_INFO(...) \
  HLLOG_INFO(PT_FALLBACK, FORMAT_AND_MSG(__VA_ARGS__));

// End of logging macros

template <
    typename Integer,
    typename = std::enable_if_t<std::is_integral<Integer>::value>>
std::string VecToString(const std::vector<Integer>& vec) {
  std::ostringstream sstr;
  sstr << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    sstr << (i > 0 ? ", " : "") << (unsigned)vec[i];
  }
  sstr << "]";
  return sstr.str();
}

#define CREATE_OSTREAM_FORMATTER(type) \
  template <>                          \
  struct fmt::formatter<type> : ostream_formatter {};
