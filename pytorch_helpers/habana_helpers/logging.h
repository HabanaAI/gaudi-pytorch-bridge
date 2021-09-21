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

#include <synapse_helpers/runtime_tracing.h>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include "pytorch_helpers/synapse_helpers/env_flags.h"
#define FMT_HEADER_ONLY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "spdlog/common.h"
#include "spdlog/fmt/bundled/format.h"
#pragma GCC diagnostic pop

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

} // namespace Logger

inline char* get2env(const char* a) {
  std::string b; // new Option
  // Get new options
  if (0 == strcmp("PT_HABANA_LOG_MOD_MASK", a))
    b = "PT_HPU_LOG_MOD_MASK";
  else if (0 == strcmp("PT_HABANA_LOG_TYPE_MASK", a))
    b = "PT_HPU_LOG_TYPE_MASK";
  else if (0 == strcmp("PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION", a))
    b = "PT_HPU_ENABLE_GRAPHMODE_LAYERNORM_FUSION";
  else if (0 == strcmp("HABANA_PGM_ENABLE_CACHE", a))
    b = "PT_HPU_PGM_ENABLE_CACHE";

  // Search
  char* mask = getenv(a);
  if (mask != nullptr) {
    return mask;
  } else {
    return getenv(b.c_str());
  }
}

class PtLogger {
 private:
  static PtLogger* instance;
  unsigned long module_mask_;
  unsigned long type_mask_;

  PtLogger() {
    char* mask = get2env("PT_HABANA_LOG_MOD_MASK");
    char* node_mask_ptr = std::getenv("PT_HPU_LOG_NODE_MASK");
    char* gc_log_level_ptr = std::getenv("PT_HPU_SYN_LOG_LEVEL");

    unsigned long node_id = 0;
    unsigned long node_id_mask = 0;

    if (node_mask_ptr != nullptr) {
      node_id_mask = std::stoul(node_mask_ptr, nullptr, 16);

      // multinode can be either rank or id
      char* node_id_ptr = std::getenv("ID");
      if (node_id_ptr != nullptr) {
        node_id = std::stoul(node_id_ptr, nullptr, 16);
      }
    }

    if (gc_log_level_ptr != nullptr) {
      if (node_mask_ptr != nullptr) {
        if ((node_id_mask & (1 << node_id)) == 1) {
          setenv("LOG_LEVEL_ALL", gc_log_level_ptr, 1);
        }
      } else {
        setenv("LOG_LEVEL_ALL", gc_log_level_ptr, 1);
      }
    }

    if (mask != nullptr) {
      module_mask_ = std::stoul(mask, nullptr, 16); // expects hex

      // retain the default mask for other nodes
      if (node_mask_ptr != nullptr) {
        if ((node_id_mask & (1 << node_id)) == 0) {
          module_mask_ = INT64_MAX;
        }
      }

    } else {
      // enable all modules by default
      module_mask_ = INT64_MAX;
    }

    mask = get2env("PT_HABANA_LOG_TYPE_MASK");
    if (mask != nullptr) {
      type_mask_ = std::stoul(mask, nullptr, 16); // expects hex

      // retain the default mask for other nodes
      if (node_mask_ptr != nullptr) {
        if ((node_id_mask & (1 << node_id)) == 0) {
          type_mask_ = TypeMask::FATAL + TypeMask::WARNING;
        }
      }
    } else {
      // enable fatal errors and warnings by default
      type_mask_ = TypeMask::FATAL + TypeMask::WARNING;
      // Always set the debug logs for lazy and bridge so that
      // we get the detailed info on the graphs etc that was launched
      // when something fails and can be reproduced manually
      if (const auto envp = std::getenv("PT_HPU_LAZY_MODE")) {
        if (3 == std::stoul(envp, nullptr, 10)) {
          type_mask_ += TypeMask::DEBUG;
        }
      }
    }
  }

 public:
  PtLogger(const PtLogger&) = delete;
  PtLogger& operator=(const PtLogger&) = delete;

  static PtLogger* getLogger() {
    if (instance == nullptr) {
      instance = new PtLogger();
    }

    return instance;
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
    FATAL = 0x1,
    WARNING = 0x2,
    TRACE = 0x4,
    DEBUG = 0x8,
    PROFILE = 0x10,
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
  };
};

class PTFuncLog {
 private:
  std::string pName;
  std::string name;
  bool isDebug;

 public:
  PTFuncLog(std::string pn, std::string n, bool debug)
      : pName(std::move(pn)), name(std::move(n)), isDebug(debug) {
    if (isDebug) {
      std::clog << "HABANA_LOG: begin of " << pName << "\n";
    }
    synapse_helpers::trace_start(name.c_str());
  }
  ~PTFuncLog() {
    if (isDebug) {
      std::clog << "HABANA_LOG: end of " << pName << "\n";
    }
    synapse_helpers::trace_end(name.c_str());
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
#define PT_MOD_FATAL(MOD, ...)                             \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) && \
       (PtLogger::getLogger()->getTypeMask() &             \
        (PtLogger::TypeMask::FATAL)))) {                   \
    Logger::habana_assert(                                 \
        __func__,                                          \
        __FILE__,                                          \
        static_cast<uint32_t>(__LINE__),                   \
        Logger::str(__VA_ARGS__));                         \
  }

#define PT_DEVICE_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DEVICE, __VA_ARGS__)

#define PT_KERNEL_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::KERNEL, __VA_ARGS__)

#define PT_BRIDGE_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::BRIDGE, __VA_ARGS__)

#define PT_SYNHELPER_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::SYNHELPER, __VA_ARGS__)

#define PT_DISTRIBUTED_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)

#define PT_LAZY_FATAL(...) PT_MOD_FATAL(PtLogger::ModuleMask::LAZY, __VA_ARGS__)

#define PT_HABANAHOOKS_FATAL(...) \
  PT_MOD_FATAL(PtLogger::ModuleMask::HABANAHOOKS, __VA_ARGS__)

/************************WARNING MACROS************************/
#define PT_MOD_WARN(MOD, ...)                                       \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&          \
       (PtLogger::getLogger()->getTypeMask() &                      \
        (PtLogger::TypeMask::WARNING)))) {                          \
    std::cerr << Logger::str(__VA_ARGS__) << " " << __FILE__ << ":" \
              << __LINE__ << "\t" << __func__ << "\n";              \
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

#define PT_DISTRIBUTED_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)

#define PT_LAZY_WARN(...) PT_MOD_WARN(PtLogger::ModuleMask::LAZY, __VA_ARGS__)

#define PT_HABANAHOOKS_WARN(...) \
  PT_MOD_WARN(PtLogger::ModuleMask::HABANAHOOKS, __VA_ARGS__)

#define PT_FALLBACK_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PtLogger::ModuleMask::FALLBACK, __VA_ARGS__)

#define PT_TEST_WARN(...) \
  PT_MOD_WARN_WITHOUT_LINE_FILE(PtLogger::ModuleMask::TEST, __VA_ARGS__)

/************************TRACE MACROS************************************/
#define PT_MOD_BEGIN(MOD)                                                \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&               \
       (PtLogger::getLogger()->getTypeMask() &                           \
        (PtLogger::TypeMask::TRACE)))) {                                 \
    std::clog << "HABANA_LOG: begin of " << __PRETTY_FUNCTION__ << "\n"; \
  };                                                                     \
  synapse_helpers::trace_start(__FUNCTION__);

#define PT_DEVICE_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::DEVICE)
#define PT_KERNEL_BEGIN                                           \
  {                                                               \
    bool lazy_mode = GET_ENV_FLAG(PT_HPU_LAZY_MODE);              \
    HABANA_ASSERT(                                                \
        !lazy_mode,                                               \
        "Lazy Mode = ",                                           \
        lazy_mode,                                                \
        "  :  "                                                   \
        "Please avoid Legacy eager calls in Lazy execution mode " \
        "(for optimizers use PT_OPTIMIZER_KERNEL_BEGIN),"         \
        " for other kernels use PT_OTHER_KERNEL_BEGIN");          \
    PT_MOD_BEGIN(PtLogger::ModuleMask::KERNEL)                    \
  }
// following macro is a non-asserting version of PT_KERNEL_BEGIN
#define PT_OTHER_OPS_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::KERNEL)
#define PT_BRIDGE_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::BRIDGE)
#define PT_SYNHELPER_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::SYNHELPER)
#define PT_DISTRIBUTED_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::DISTRIBUTED)
#define PT_LAZY_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::LAZY)

#define PT_MOD_END(MOD)                                                \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&             \
       (PtLogger::getLogger()->getTypeMask() &                         \
        (PtLogger::TypeMask::TRACE)))) {                               \
    std::clog << "HABANA_LOG: end of " << __PRETTY_FUNCTION__ << "\n"; \
  };                                                                   \
  synapse_helpers::trace_end(__FUNCTION__);

#define PT_DEVICE_END PT_MOD_END(PtLogger::ModuleMask::DEVICE)
#define PT_KERNEL_END PT_MOD_END(PtLogger::ModuleMask::KERNEL)
#define PT_BRIDGE_END PT_MOD_END(PtLogger::ModuleMask::BRIDGE)
#define PT_SYNHELPER_END PT_MOD_END(PtLogger::ModuleMask::SYNHELPER)
#define PT_DISTRIBUTED_END PT_MOD_END(PtLogger::ModuleMask::DISTRIBUTED)
#define PT_LAZY_END PT_MOD_END(PtLogger::ModuleMask::LAZY)

#define PT_MOD_TRACE(MOD, PNAME, NAME)                     \
  bool isDebug = false;                                    \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) && \
       (PtLogger::getLogger()->getTypeMask() &             \
        (PtLogger::TypeMask::TRACE)))) {                   \
    isDebug = true;                                        \
  };                                                       \
  PTFuncLog ptFuncLogger(PNAME, NAME, isDebug);

#define PT_LAZY_TRACE \
  PT_MOD_TRACE(PtLogger::ModuleMask::LAZY, __PRETTY_FUNCTION__, __FUNCTION__)

#define PT_FALLBACK_TRACE \
  PT_MOD_TRACE(           \
      PtLogger::ModuleMask::FALLBACK, __PRETTY_FUNCTION__, __FUNCTION__)
#define PT_SYNHELPER_TRACE \
  PT_MOD_TRACE(            \
      PtLogger::ModuleMask::SYNHELPER, __PRETTY_FUNCTION__, __FUNCTION__)

/************************DEBUG MACROS************************************/
#define PT_MOD_DEBUG(MOD, ...)                             \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) && \
       (PtLogger::getLogger()->getTypeMask() &             \
        (PtLogger::TypeMask::DEBUG)))) {                   \
    std::clog << Logger::str(__VA_ARGS__) << "\n";         \
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
#define PT_SYNHELPER_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::SYNHELPER, __VA_ARGS__)
#define PT_DISTRIBUTED_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::DISTRIBUTED, __VA_ARGS__)
#define PT_LAZY_DEBUG(...) PT_MOD_DEBUG(PtLogger::ModuleMask::LAZY, __VA_ARGS__)
#define PT_HABANAHOOKS_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::HABANAHOOKS, __VA_ARGS__)
#define PT_FALLBACK_DEBUG(...) \
  PT_MOD_DEBUG(PtLogger::ModuleMask::FALLBACK, __VA_ARGS__)
#define PT_TEST_DEBUG(...) PT_MOD_DEBUG(PtLogger::ModuleMask::TEST, __VA_ARGS__)