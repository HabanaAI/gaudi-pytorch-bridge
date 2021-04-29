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
} // namespace Logger

class PtLogger {
 private:
  static PtLogger* instance;
  unsigned long module_mask_;
  unsigned long type_mask_;

  PtLogger() {
    char* mask = getenv("PT_HABANA_LOG_MOD_MASK");
    if (mask != nullptr) {
      module_mask_ = std::stoul(mask, nullptr, 16); // expects hex
    } else {
      // enable all modules by default
      module_mask_ = INT64_MAX;
    }

    mask = getenv("PT_HABANA_LOG_TYPE_MASK");
    if (mask != nullptr) {
      type_mask_ = std::stoul(mask, nullptr, 16); // expects hex
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

  enum TypeMask {
    FATAL = 0x1,
    WARNING = 0x2,
    TRACE = 0x4,
    DEBUG = 0x8,
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
  };
};

class PTFuncLog {
 private:
  std::string pName;
  std::string name;
  bool isDebug;

 public:
  PTFuncLog(const std::string& pn, const std::string& n, bool debug)
      : pName(pn), name(n), isDebug(debug) {
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

/************************CRITICAL MACROS************************/
#define PT_MOD_FATAL(MOD, ...)                                      \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&          \
       (PtLogger::getLogger()->getTypeMask() &                      \
        (PtLogger::TypeMask::FATAL)))) {                            \
    std::cerr << Logger::str(__VA_ARGS__) << " " << __FILE__ << ":" \
              << __LINE__ << "\t" << __func__ << "\n";              \
    std::terminate();                                               \
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

#define HABANA_ASSERT(condition)                                             \
  {                                                                          \
    if (!(condition)) {                                                      \
      std::cerr << "Assertion (" << #condition << ") is false! " << __FILE__ \
                << ":" << __LINE__ << "\t" << __func__ << "\n";              \
      std::terminate();                                                      \
    }                                                                        \
  }

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

/************************TRACE MACROS************************************/
#define PT_MOD_BEGIN(MOD)                                                \
  if (((PtLogger::getLogger()->getModuleMask() & (MOD)) &&               \
       (PtLogger::getLogger()->getTypeMask() &                           \
        (PtLogger::TypeMask::TRACE)))) {                                 \
    std::clog << "HABANA_LOG: begin of " << __PRETTY_FUNCTION__ << "\n"; \
  };                                                                     \
  synapse_helpers::trace_start(__FUNCTION__);

#define PT_DEVICE_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::DEVICE)
#define PT_KERNEL_BEGIN PT_MOD_BEGIN(PtLogger::ModuleMask::KERNEL)
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
