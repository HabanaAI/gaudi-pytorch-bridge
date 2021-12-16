
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
#pragma once
#include <cxxabi.h>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#define FMT_HEADER_ONLY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "spdlog/common.h"
#include "spdlog/fmt/bundled/format.h"
#pragma GCC diagnostic pop

#include <sys/syscall.h> // For syscall(__NR_gettid)
#include <unistd.h> // For syscall(__NR_gettid)

#ifdef WIN32
#define __FILENAME__ \
  (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

namespace spdlog {
class logger;
};

namespace ptspdlogger {
class LogManager {
 public:
  enum class LogType : uint32_t {
    DEVICE, /*0x1*/
    KERNEL, /*0x2*/
    BRIDGE, /*0x4*/
    SYNHELPER, /*0x8*/
    DISTRIBUTED, /*0x10*/
    LAZY, /*0x20*/
    HABANAHOOKS, /*0x40*/
    FALLBACK, /*0x80*/
    STATS, /*0x100*/
    TEST, /*0x200*/
    DYNAMIC_SHAPE, /*0x400*/
    DEVMEM, /*0x800*/
    HABHELPER, /*0x1000*/

    LOG_MAX // Must be last
  };

  static LogManager& instance();

  ~LogManager();

  void create_logger(const LogType& logType);

  void drop_log(const LogType& logType);

  void set_log_level(const LogType& logType, unsigned log_level);

  unsigned get_log_level(const LogType& logType);

  void set_logger_sink(
      const LogType& logType,
      const std::string& pathname,
      unsigned lvl,
      size_t size,
      size_t amount);

  template <typename... Args>
  void log(
      const LogType& logType,
      const int logLevel,
      const char* s,
      const char* file,
      int line,
      const Args&... args) {
    std::string prefix = m_printFileAndLine
        ? std::string(file) + "::" + std::to_string(line) + " "
        : "";

    log_wrapper(logType, logLevel, prefix + fmt::format(s, args...));
  }

  void clearLogContext();

  void setLogPattern();

  void* enableConsole(const LogManager::LogType& logType);

  bool disableConsole(const LogManager::LogType& logType, void* console);

 private:
  LogManager();

  typedef std::shared_ptr<spdlog::logger> LoggerPtr;
  typedef std::array<LoggerPtr, (uint32_t)LogType::LOG_MAX> LoggersMap;
  typedef std::array<int, (uint32_t)LogType::LOG_MAX> LoggerLevels;
  typedef std::set<LogType> SeparateLogTypes;

  LoggerPtr getLogger(const LogType& logType, const std::string& msg) const;

  void log_wrapper(const LogType& logType, const int logLevel, std::string&& s);

  const std::string& getLogTypeString(const LogType& logType) const;

  unsigned long getLogMask(const LogManager::LogType& logType) const;

  LogManager::LogType getLogType(const unsigned logTypeMask) const;

  LoggersMap m_loggersMap;
  std::string m_logPattern;
  bool m_printFileAndLine;
  LoggerLevels m_loggerLevels;
  std::mutex m_mutex;
  SeparateLogTypes m_sepLogTypes;

  const int m_env_log_level_all;
  int m_env_log_module_mask; // Each bit represents a module
  const unsigned long m_env_log_type; // Log level for module mask
};

// Class for scoped log objects. The object will log at the creation and
// destruction execution only.
class FuncScopeLog {
 public:
  FuncScopeLog(const std::string& function);

  ~FuncScopeLog();

 private:
  const std::string m_function;
};

} // namespace ptspdlogger

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

template <int LEVEL>
inline bool log_level_at_least(
    const ptspdlogger::LogManager::LogType& logType) {
  return ptspdlogger::LogManager::instance().get_log_level(logType) <= LEVEL;
}

inline bool log_level_at_least(
    const ptspdlogger::LogManager::LogType& logType,
    uint32_t level) {
  return ptspdlogger::LogManager::instance().get_log_level(logType) <= level;
}

#define LOG_LEVEL_AT_LEAST_TRACE(log_type) \
  (log_level_at_least<0>(ptspdlogger::LogManager::LogType::log_type))
#define LOG_LEVEL_AT_LEAST_DEBUG(log_type) \
  (log_level_at_least<1>(ptspdlogger::LogManager::LogType::log_type))
#define LOG_LEVEL_AT_LEAST_INFO(log_type) \
  (log_level_at_least<2>(ptspdlogger::LogManager::LogType::log_type))
#define LOG_LEVEL_AT_LEAST_WARN(log_type) \
  (log_level_at_least<3>(ptspdlogger::LogManager::LogType::log_type))
#define LOG_LEVEL_AT_LEAST_ERR(log_type) \
  (log_level_at_least<4>(ptspdlogger::LogManager::LogType::log_type))
#define LOG_LEVEL_AT_LEAST_CRITICAL(log_type) \
  (log_level_at_least<5>(ptspdlogger::LogManager::LogType::log_type))

#define LOG_LEVEL_CHECK(log_type, level, OP)                \
  {                                                         \
    if (unlikely(log_level_at_least<level>(                 \
            ptspdlogger::LogManager::LogType::log_type))) { \
      OP;                                                   \
    }                                                       \
  }
#define SYN_LOG(log_type, loglevel, msg, ...) \
  ptspdlogger::LogManager::instance().log(    \
      log_type, loglevel, msg, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define SYN_LOG_TYPE(log_type, loglevel, msg, ...) \
  SYN_LOG(                                         \
      ptspdlogger::LogManager::LogType::log_type,  \
      loglevel,                                    \
      msg,                                         \
      ##__VA_ARGS__)

#define LOG_TRACE(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 0, SYN_LOG_TYPE(log_type, 0, msg, ##__VA_ARGS__));
#define LOG_DEBUG(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 1, SYN_LOG_TYPE(log_type, 1, msg, ##__VA_ARGS__));
#define LOG_INFO(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 2, SYN_LOG_TYPE(log_type, 2, msg, ##__VA_ARGS__));
#define LOG_WARN(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 3, SYN_LOG_TYPE(log_type, 3, msg, ##__VA_ARGS__));
#define LOG_ERR(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 4, SYN_LOG_TYPE(log_type, 4, msg, ##__VA_ARGS__));
#define LOG_CRITICAL(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 5, SYN_LOG_TYPE(log_type, 5, msg, ##__VA_ARGS__));

#define LOG_TRACE_T(log_type, msg, ...) \
  LOG_LEVEL_CHECK(                      \
      log_type,                         \
      0,                                \
      SYN_LOG_TYPE(                     \
          log_type, 0, "tid {} " msg, syscall(__NR_gettid), ##__VA_ARGS__));
#define LOG_DEBUG_T(log_type, msg, ...) \
  LOG_LEVEL_CHECK(                      \
      log_type,                         \
      1,                                \
      SYN_LOG_TYPE(                     \
          log_type, 1, "tid {} " msg, syscall(__NR_gettid), ##__VA_ARGS__));
#define LOG_INFO_T(log_type, msg, ...) \
  LOG_LEVEL_CHECK(                     \
      log_type,                        \
      2,                               \
      SYN_LOG_TYPE(                    \
          log_type, 2, "tid {} " msg, syscall(__NR_gettid), ##__VA_ARGS__));
#define LOG_WARN_T(log_type, msg, ...) \
  LOG_LEVEL_CHECK(                     \
      log_type,                        \
      3,                               \
      SYN_LOG_TYPE(                    \
          log_type, 3, "tid {} " msg, syscall(__NR_gettid), ##__VA_ARGS__));
#define LOG_ERR_T(log_type, msg, ...) \
  LOG_LEVEL_CHECK(                    \
      log_type,                       \
      4,                              \
      SYN_LOG_TYPE(                   \
          log_type, 4, "tid {} " msg, syscall(__NR_gettid), ##__VA_ARGS__));
#define LOG_CRITICAL_T(log_type, msg, ...) \
  LOG_LEVEL_CHECK(                         \
      log_type,                            \
      5,                                   \
      SYN_LOG_TYPE(                        \
          log_type, 5, "tid {} " msg, syscall(__NR_gettid), ##__VA_ARGS__));

#define STATIC_LOG_TRACE(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 0, SYN_LOG_TYPE(log_type, 0, msg, ##__VA_ARGS__));
#define STATIC_LOG_DEBUG(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 1, SYN_LOG_TYPE(log_type, 1, msg, ##__VA_ARGS__));
#define STATIC_LOG_INFO(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 2, SYN_LOG_TYPE(log_type, 2, msg, ##__VA_ARGS__));
#define STATIC_LOG_WARN(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 3, SYN_LOG_TYPE(log_type, 3, msg, ##__VA_ARGS__));
#define STATIC_LOG_ERR(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 4, SYN_LOG_TYPE(log_type, 4, msg, ##__VA_ARGS__));
#define STATIC_LOG_CRITICAL(log_type, msg, ...) \
  LOG_LEVEL_CHECK(log_type, 5, SYN_LOG_TYPE(log_type, 5, msg, ##__VA_ARGS__));

#define SET_LOGGER_SINK(log_type, pathname, lvl, size, amount) \
  ptspdlogger::LogManager::instance().set_logger_sink(         \
      ptspdlogger::LogManager::LogType::log_type,              \
      pathname,                                                \
      lvl,                                                     \
      size,                                                    \
      amount);
#define CREATE_LOGGER(log_type, fileName, logFileSize, logFileAmount) \
  ptspdlogger::LogManager::instance().create_logger(                  \
      ptspdlogger::LogManager::LogType::log_type,                     \
      fileName,                                                       \
      logFileSize,                                                    \
      logFileAmount);
#define DROP_LOGGER(log_type)                   \
  ptspdlogger::LogManager::instance().drop_log( \
      ptspdlogger::LogManager::LogType::log_type);

#define LOG_FUNC_SCOPE() ptspdlogger::FuncScopeLog log(__FUNCTION__)

#define TO64(x) ((uint64_t)x)
#define TO64P(x) ((void*)x)

#ifdef ENFORCE_TRACE_MODE_LOGGING
extern bool g_traceModeLogging;
#define TURN_ON_TRACE_MODE_LOGGING() g_traceModeLogging = true
#define TURN_OFF_TRACE_MODE_LOGGING() g_traceModeLogging = false
#else
const bool g_traceModeLogging = false;
#endif
