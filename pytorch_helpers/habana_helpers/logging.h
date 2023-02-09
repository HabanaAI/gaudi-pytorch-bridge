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

#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include "backend/profiling/profiling.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/runtime_tracing.h"
#define FMT_HEADER_ONLY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>
#pragma GCC diagnostic pop
#include <absl/strings/str_format.h>

// Redefining c10 StringUtils functions here as distributed and syn
// helpers are independent of  torch libraries
namespace Logger {
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

void habana_assert(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg);

template <class... Args>
inline void nop(__attribute__((unused)) const Args&... args){};
} // namespace Logger

class PtLogger {
 private:
  static PtLogger* instance;
  unsigned long module_mask_;
  unsigned long type_mask_;

  void loadMask() {
    module_mask_ = GET_ENV_FLAG_NEW(PT_HPU_LOG_MOD_MASK);
    type_mask_ = GET_ENV_FLAG_NEW(PT_HPU_LOG_TYPE_MASK);

    char* gc_log_level_ptr = std::getenv("PT_HPU_SYN_LOG_LEVEL");

    unsigned long node_id = 0;
    unsigned long node_id_mask = GET_ENV_FLAG_NEW(PT_HPU_LOG_NODE_MASK);

    if (node_id_mask) {
      char* node_id_ptr = std::getenv("RANK");
      if (node_id_ptr != nullptr) {
        node_id = std::stoul(node_id_ptr, nullptr, 10);
      }
    }

    if (gc_log_level_ptr != nullptr) {
      if (node_id_mask) {
        if ((node_id_mask & (1 << node_id)) == 1) {
          setenv("LOG_LEVEL_ALL", gc_log_level_ptr, 1);
        }
      } else {
        setenv("LOG_LEVEL_ALL", gc_log_level_ptr, 1);
      }
    }

    // retain the default mask for other nodes
    if (module_mask_) {
      if (node_id_mask) {
        if ((node_id_mask & (1 << node_id)) == 0) {
          module_mask_ = UINT64_MAX;
        }
      }
    }
    // retain the default mask for other nodes
    if (type_mask_) {
      if (node_id_mask) {
        if ((node_id_mask & (1 << node_id)) == 0) {
          type_mask_ = TypeMask::WARNING;
        }
      }
    } else {
      // Always set the debug logs for lazy and bridge so that
      // we get the detailed info on the graphs etc that was launched
      // when something fails and can be reproduced manually
      if (3 == GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE)) {
        type_mask_ += TypeMask::DEBUG;
      }
    }
  }
  PtLogger();

 public:
  PtLogger(const PtLogger&) = delete;
  PtLogger& operator=(const PtLogger&) = delete;

  static PtLogger* getLogger() {
    if (instance == nullptr) {
      instance = new PtLogger();
    }

    return instance;
  }

  spdlog::logger* GetOpLogger() {
    return spdlog::get("PYTORCH_HPU_OPS").get();
  }

  void refresh() {
    loadMask();
  }

  unsigned long getTypeMask() {
    return type_mask_;
  }

  unsigned long getModuleMask() {
    return module_mask_;
  }

  void moduleMaskOr(unsigned long toggle_on) {
    module_mask_ |= toggle_on;
    return;
  }

  void typeMaskOr(unsigned long toggle_on) {
    type_mask_ |= toggle_on;
    return;
  }

  enum TypeMask {
    WARNING = 0x1,
    TRACE = 0x2,
    DEBUG = 0x4,
    PROFILE = 0x8,
    RUNTIME_PROFILE = 0x10,
    TENSORBOARD = 0x20
  };

  enum ModuleMask {
    DEVICE = 0x1,
    KERNEL = 0x2,
    BRIDGE = 0x4,
    SYNHELPER = 0x8,
    DISTRIBUTED = 0x10,
    LAZY = 0x20,
    HABANAHOOKS = 0x40,
    FALLBACK = 0x80,
    STATS = 0x100,
    TEST = 0x200,
    DYNAMIC_SHAPE = 0x400,
    DEVMEM = 0x800,
    HABHELPER = 0x1000,
    IRGRAPH = 0x2000,
    VIEWTABLE = 0x4000,
    REFINEMENT = 0x8000,
    HOSTSTAT = 0x10000,
    LAYOUTS = 0x20000,
    PARALLEL_ACC = 0x40000,
    LAZY_EAGER = 0x80000,
    MEMLOG = 0x100000,
    EXEC_THREAD = 0x200000,
    EAGER = 0x400000,
    CUSTOM = 0x800000 // Don't use it in checkin code.
  };
};

namespace Logger {
inline std::string DebugString(const PtLogger::ModuleMask& mod) {
  switch (mod) {
    case PtLogger::ModuleMask::DEVICE:
      return std::string("DEVICE");
    case PtLogger::ModuleMask::KERNEL:
      return std::string("KERNEL");
    case PtLogger::ModuleMask::BRIDGE:
      return std::string("BRIDGE");
    case PtLogger::ModuleMask::SYNHELPER:
      return std::string("SYNHELPER");
    case PtLogger::ModuleMask::DISTRIBUTED:
      return std::string("DISTRIBUTED");
    case PtLogger::ModuleMask::LAZY:
      return std::string("LAZY");
    case PtLogger::ModuleMask::HABANAHOOKS:
      return std::string("HABANAHOOKS");
    case PtLogger::ModuleMask::FALLBACK:
      return std::string("FALLBACK");
    case PtLogger::ModuleMask::STATS:
      return std::string("STATS");
    case PtLogger::ModuleMask::TEST:
      return std::string("TEST");
    case PtLogger::ModuleMask::DYNAMIC_SHAPE:
      return std::string("DYNAMIC_SHAPE");
    case PtLogger::ModuleMask::DEVMEM:
      return std::string("DEVMEM");
    case PtLogger::ModuleMask::HOSTSTAT:
      return std::string("HOSTSTAT");
    case PtLogger::ModuleMask::HABHELPER:
      return std::string("HABHELPER");
    case PtLogger::ModuleMask::PARALLEL_ACC:
      return std::string("PARALLEL_ACC");
    case PtLogger::ModuleMask::LAZY_EAGER:
      return std::string("LAZY_EAGER");
    case PtLogger::ModuleMask::MEMLOG:
      return std::string("MEMLOG");
    case PtLogger::ModuleMask::EXEC_THREAD:
      return std::string("EXEC_THREAD");
    case PtLogger::ModuleMask::EAGER:
      return std::string("EAGER");
    case PtLogger::ModuleMask::CUSTOM:
      return std::string("CUSTOM");
    default:
      return std::string("UNDEFINED");
  }
}
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
      absl::StrAppend(&message, module, ": begin of ", pName, "\n");
      std::clog << message;
    }
    synapse_helpers::trace_start(name.data());
    habana::profile::bridge::trace_start(name);
  }
  ~PTFuncLog() {
    if (isActive) {
      auto message{Logger::print_hdr()};
      absl::StrAppend(&message, module, ": end of ", pName, "\n");
      std::clog << message;
    }
    synapse_helpers::trace_end(name.data());
    habana::profile::bridge::trace_end(name);
  }
};

#define HABANA_CHECK_MSG(cond, ...) \
  Logger::CheckMsgImpl(             \
      "Expected " #cond " to be true, but got false.", ##__VA_ARGS__)

#define HABANA_ASSERT(condition, ...)                         \
  if (__builtin_expect(static_cast<bool>(!(condition)), 0)) { \
    Logger::habana_assert(                                    \
        __func__,                                             \
        __FILE__,                                             \
        static_cast<uint32_t>(__LINE__),                      \
        HABANA_CHECK_MSG(condition, ##__VA_ARGS__));          \
  }

/************************CRITICAL MACROS************************/
#define PT_MOD_FATAL(MOD, ...)                               \
  {                                                          \
    Logger::habana_assert(                                   \
        __func__,                                            \
        __FILE__,                                            \
        static_cast<uint32_t>(__LINE__),                     \
        Logger::str(                                         \
            Logger::print_hdr() + "FATAL ERROR :: MODULE:" + \
            Logger::DebugString(MOD) + " " + __VA_ARGS__));  \
  }

#define PT_DEVICE_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DEVICE, __VA_ARGS__)

#define PT_KERNEL_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::KERNEL, __VA_ARGS__)

#define PT_BRIDGE_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::BRIDGE, __VA_ARGS__)

#define PT_SYNHELPER_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::SYNHELPER, __VA_ARGS__)

#define PT_HABHELPER_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::HABHELPER, __VA_ARGS__)

#define PT_DEVMEM_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DEVMEM, __VA_ARGS__)

#define PT_DISTRIBUTED_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)

#define PT_LAZY_FATAL(...) PT_MOD_FATAL(PtLogger::ModuleMask::LAZY, __VA_ARGS__)

#define PT_IRGRAPH_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::IRGRAPH, __VA_ARGS__)

#define PT_HABANAHOOKS_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::HABANAHOOKS, __VA_ARGS__)

#define PT_DYNAMIC_SHAPE_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DYNAMIC_SHAPE, __VA_ARGS__)

#define PT_LAZY_EAGER_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::LAZY_EAGER, __VA_ARGS__)

/************************WARNING MACROS************************/
#define PT_MOD_WARN(MOD, ...)                                           \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&              \
       (PtLogger::getLogger()->getTypeMask() &                          \
        (PtLogger::TypeMask::WARNING)))) {                              \
    Logger::print(std::cerr, __VA_ARGS__);                              \
    std::cerr << " " << __FILE__ << ":" << __LINE__ << "\t" << __func__ \
              << "\n";                                                  \
  }

#define PT_MOD_WARN_WITHOUT_LINE_FILE(MOD, ...)            \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) && \
       (PtLogger::getLogger()->getTypeMask() &             \
        (PtLogger::TypeMask::WARNING)))) {                 \
    std::cerr << Logger::str(__VA_ARGS__) << "\n";         \
  }

#define PT_DEVICE_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::DEVICE, __VA_ARGS__)

#define PT_KERNEL_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::KERNEL, __VA_ARGS__)

#define PT_BRIDGE_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::BRIDGE, __VA_ARGS__)

#define PT_SYNHELPER_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::SYNHELPER, __VA_ARGS__)

#define PT_HABHELPER_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::HABHELPER, __VA_ARGS__)

#define PT_DEVMEM_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::DEVMEM, __VA_ARGS__)

#define PT_DISTRIBUTED_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)

#define PT_LAZY_WARN(...) PT_MOD_WARN(PtLogger::ModuleMask::LAZY, __VA_ARGS__)

#define PT_IRGRAPH_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::IRGRAPH, __VA_ARGS__)

#define PT_HABANAHOOKS_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::HABANAHOOKS, __VA_ARGS__)

#define PT_FALLBACK_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PtLogger::ModuleMask::FALLBACK, __VA_ARGS__)

#define PT_TEST_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PtLogger::ModuleMask::TEST, __VA_ARGS__)

#define PT_DYNAMIC_SHAPE_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(   \
      PtLogger::ModuleMask::DYNAMIC_SHAPE, __VA_ARGS__)

#define PT_LAZY_EAGER_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PtLogger::ModuleMask::LAZY_EAGER, __VA_ARGS__)

/************************TRACE MACROS************************************/
#define PT_MOD_BEGIN(MOD) PT_MOD_SCOPE(MOD, __PRETTY_FUNCTION__, __FUNCTION__)

#define PT_DEVICE_BEGIN PT_MOD_BEGIN(DEVICE)
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
    PT_MOD_BEGIN(KERNEL)                                          \
  }
// following macro is a non-asserting version of PT_KERNEL_BEGIN
#define PT_OTHER_OPS_BEGIN PT_MOD_BEGIN(KERNEL)
#define PT_BRIDGE_BEGIN PT_MOD_BEGIN(BRIDGE)
#define PT_SYNHELPER_BEGIN PT_MOD_BEGIN(SYNHELPER)
#define PT_HABHELPER_BEGIN PT_MOD_BEGIN(HABHELPER)
#define PT_DEVMEM_BEGIN PT_MOD_BEGIN(DEVMEM)
#define PT_DISTRIBUTED_BEGIN PT_MOD_BEGIN(DISTRIBUTED)
#define PT_LAZY_BEGIN PT_MOD_BEGIN(LAZY)

#define PT_MOD_END(MOD)

#define PT_DEVICE_END PT_MOD_END(DEVICE)
#define PT_KERNEL_END PT_MOD_END(KERNEL)
#define PT_OTHER_OPS_END PT_MOD_END(KERNEL)
#define PT_BRIDGE_END PT_MOD_END(BRIDGE)
#define PT_SYNHELPER_END PT_MOD_END(SYNHELPER)
#define PT_HABHELPER_END PT_MOD_END(HABHELPER)
#define PT_DEVMEM_END PT_MOD_END(DEVMEM)
#define PT_DISTRIBUTED_END PT_MOD_END(DISTRIBUTED)
#define PT_LAZY_END PT_MOD_END(LAZY)

#define PT_MOD_SCOPE(MOD, PNAME, NAME)                    \
  std::optional<PTFuncLog> ptFuncLogger{};                \
  {                                                       \
    auto type_mask{PtLogger::getLogger()->getTypeMask()}; \
    if (type_mask > PtLogger::TypeMask::WARNING) {        \
      bool isDebug{                                       \
          (PtLogger::getLogger()->getModuleMask() &       \
           (PtLogger::ModuleMask::MOD)) &&                \
          (type_mask & (PtLogger::TypeMask::TRACE))};     \
      ptFuncLogger.emplace(#MOD, PNAME, NAME, isDebug);   \
    }                                                     \
  }

#define PT_MOD_TRACE(MOD, PNAME, NAME) PT_MOD_SCOPE(MOD, PNAME, NAME)

#define PT_EAGER_TRACE PT_MOD_TRACE(EAGER, __PRETTY_FUNCTION__, __FUNCTION__)

#define PT_LAZY_TRACE PT_MOD_TRACE(LAZY, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_LAZY_TRACE_WITH_NAME(name) PT_MOD_TRACE(LAZY, name, name)
#define PT_BRIDGE_TRACE PT_MOD_TRACE(BRIDGE, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_FALLBACK_TRACE \
  PT_MOD_TRACE(FALLBACK, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_SYNHELPER_TRACE \
  PT_MOD_TRACE(SYNHELPER, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_HABHELPER_TRACE \
  PT_MOD_TRACE(HABHELPER, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_TEST_TRACE PT_MOD_TRACE(TEST, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_DYNAMIC_SHAPE_TRACE \
  PT_MOD_TRACE(DYNAMIC_SHAPE, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_DEVMEM_TRACE PT_MOD_TRACE(DEVMEM, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_LAZY_EAGER_TRACE \
  PT_MOD_TRACE(LAZY_EAGER, __PRETTY_FUNCTION__, __FUNCTION__)

/************************DEBUG MACROS************************************/
#define IS_MOD_DEBUG_ENABLED(MOD)                      \
  ((PtLogger::getLogger()->getModuleMask() & (MOD)) && \
   (PtLogger::getLogger()->getTypeMask() & (PtLogger::TypeMask::DEBUG)))

#define PT_MOD_DEBUG(MOD, ...)                                             \
  if (IS_MOD_DEBUG_ENABLED(MOD)) {                                         \
    Logger::print(std::clog, Logger::DebugString(MOD), ": ", __VA_ARGS__); \
    std::clog << "\n";                                                     \
  };

#define PT_PROFILE_DUMP(...)                            \
  if ((PtLogger::getLogger()->getModuleMask() &         \
           (PtLogger::ModuleMask::STATS) &&             \
       (PtLogger::getLogger()->getTypeMask() &          \
        (PtLogger::TypeMask::PROFILE)))) {              \
    std::clog << fmt::format(__VA_ARGS__) << std::endl; \
  };

#define PT_DEVICE_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::DEVICE, __VA_ARGS__)
#define PT_KERNEL_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::KERNEL, __VA_ARGS__)
#define PT_BRIDGE_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::BRIDGE, __VA_ARGS__)
#define IS_BRIDGE_DEBUG_ENABLED \
  IS_MOD_DEBUG_ENABLED(PtLogger::ModuleMask::BRIDGE)
#define PT_SYNHELPER_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::SYNHELPER, __VA_ARGS__)
#define IS_SYNHELPER_DEBUG_ENABLED \
  IS_MOD_DEBUG_ENABLED(PtLogger::ModuleMask::SYNHELPER)
#define PT_HABHELPER_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::HABHELPER, __VA_ARGS__)
#define PT_DEVMEM_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::DEVMEM, __VA_ARGS__)
#define PT_DISTRIBUTED_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)
#define PT_LAZY_DEBUG(...) PT_MOD_DEBUG(PtLogger::ModuleMask::LAZY, __VA_ARGS__)
#define PT_EAGER_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::EAGER, __VA_ARGS__)
#define PT_LAZY_PARALLEL_ACC_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::PARALLEL_ACC, __VA_ARGS__)
#define PT_MEMLOG_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::MEMLOG, __VA_ARGS__)
#define IS_MEMLOG_DEBUG_ENABLED \
  IS_MOD_DEBUG_ENABLED(PtLogger::ModuleMask::MEMLOG)
#define PT_IRGRAPH_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::IRGRAPH, __VA_ARGS__)
#define PT_VIEWTABLE_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::VIEWTABLE, __VA_ARGS__)
#define PT_HOSTSTAT_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::HOSTSTAT, __VA_ARGS__)
#define PT_HABANAHOOKS_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::HABANAHOOKS, __VA_ARGS__)
#define PT_FALLBACK_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::FALLBACK, __VA_ARGS__)
#define PT_TEST_DEBUG(...) PT_MOD_DEBUG(PtLogger::ModuleMask::TEST, __VA_ARGS__)
#define PT_DYNAMIC_SHAPE_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::DYNAMIC_SHAPE, __VA_ARGS__)
#define PT_REFINEMENT_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::REFINEMENT, __VA_ARGS__)
#define PT_LAYOUTS_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::LAYOUTS, __VA_ARGS__)
#define PT_LAZY_EAGER_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::LAZY_EAGER, __VA_ARGS__)
#define PT_LAZY_EXEC_THREAD(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::EXEC_THREAD, __VA_ARGS__)
#define PT_CUSTOM_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::CUSTOM, __VA_ARGS__)

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

#define PT_OP_INFO(...)                                        \
  {                                                            \
    const auto& logger = PtLogger::getLogger()->GetOpLogger(); \
    if (logger->should_log(spdlog::level::info)) {             \
      logger->info("{}", c10::str(__VA_ARGS__));               \
    }                                                          \
  }
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
