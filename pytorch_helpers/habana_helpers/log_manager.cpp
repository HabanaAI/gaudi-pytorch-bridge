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
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include "pytorch_helpers/synapse_helpers/env_flags.h"
// clang-format off
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <spdlog/spdlog.h>
#include <spdlog/details/fmt_helper.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#pragma GCC diagnostic pop
// clang-format on
#include "log_manager.h"
#ifdef _WIN32
#define COLOR_MACRO spdlog::sinks::wincolor_stdout_sink_mt
#else
#define COLOR_MACRO spdlog::sinks::ansicolor_stdout_sink_mt
#endif
#define LOGGER_NAME_MAX_LENGTH 15
#define SPDLOG_TRACE_ON

#define NUM_OF_ELEMENTS(array) sizeof(array) / sizeof(array[0])
#define COMPILE_TIME_ASSERT_VERIFY_ARRAY_SIZE(array, numOfElements) \
  static_assert(NUM_OF_ELEMENTS(array) == numOfElements, "array size mismatch ")

#define LOG_SIZE 10 * 1024 * 1024
#define USER_LOG_LEVEL 2 // spdlog::level::info
#define DEVICE_LOG_MASK 0x1
#define KERNEL_LOG_MASK 0x2
#define BRIDGE_LOG_MASK 0x4
#define SYNHELPER_LOG_MASK 0x8
#define DISTRIBUTED_LOG_MASK 0x10
#define LAZY_LOG_MASK 0x20
#define HABANAHOOKS_LOG_MASK 0x40
#define FALLBACK_LOG_MASK 0x80
#define STATS_LOG_MASK 0x100
#define TEST_LOG_MASK 0x200
#define DYNAMIC_SHAPE_LOG_MASK 0x400
#define DEVMEM_LOG_MASK 0x800
#define HABHELPER_LOG_MASK 0x1000
#define NOT_SUPPORTED_LOG_MASK 0xcafe

thread_local static std::string s_logContext;
thread_local static std::deque<std::string> s_logContextStack;
thread_local static spdlog::sink_ptr s_multiGraphLogSink;
thread_local static int s_threadIdx = -1;
static int s_numThreads = 0;

namespace ptspdlogger {

inline std::shared_ptr<spdlog::logger> validate(
    const std::string& logname,
    const std::string& msg) {
  auto log = spdlog::get(logname);
  if (log == nullptr) {
    log = spdlog::get("default");
    if (log == nullptr) {
      log = spdlog::stdout_color_mt("default");
    }
    log->critical(
        "logger with name - {}, does not exist [on msg \"{}\"]", logname, msg);
    return log;
  } else {
    return log;
  }
}

static std::shared_ptr<spdlog::logger> define_logger(
    const std::string& logname) {
  std::vector<spdlog::sink_ptr> sinks;
  const char* enable_console = "true"; // getenv("ENABLE_CONSOLE");
  bool should_enable_console =
      enable_console && !strcmp(enable_console, "true");
  if (should_enable_console) {
    sinks.push_back(std::make_shared<COLOR_MACRO>());
  }
  auto combined_logger =
      std::make_shared<spdlog::logger>(logname, begin(sinks), end(sinks));
  spdlog::register_logger(combined_logger);
  if (logname == "PERF") {
    combined_logger->flush_on(spdlog::level::warn);
  } else {
    combined_logger->flush_on(spdlog::level::trace);
  }

  return combined_logger;
}

LogManager& LogManager::instance() {
  static LogManager instance;

  return instance;
}

LogManager::LogManager()
    : m_env_log_level_all(GET_ENV_FLAG_NEW(PT_HPU_LOG_LEVEL_ALL)),
      m_env_log_type(GET_ENV_FLAG_NEW(PT_HPU_LOG_TYPE)) {
  if (getenv("PT_HPU_LOG_MOD_MASK") != nullptr) {
    m_env_log_module_mask = GET_ENV_FLAG(PT_HPU_LOG_MOD_MASK);
  } else {
    m_env_log_module_mask = 0;
  }

  setLogPattern();

  std::fill(m_loggerLevels.begin(), m_loggerLevels.end(), spdlog::level::off);

  for (uint32_t logType = 0; logType < (uint32_t)LogType::LOG_MAX; logType++) {
    create_logger((LogType)logType);
  }
}

LogManager::~LogManager() {
  for (size_t i = 0; i < m_loggersMap.size(); ++i) {
    if (m_loggersMap[i] != nullptr) {
      getLogger((LogType)i, "drop_log")->flush();
    }
  }
}

void LogManager::create_logger(const LogManager::LogType& logType) {
  auto& logName = getLogTypeString(logType);
  assert(logName.length() <= LOGGER_NAME_MAX_LENGTH);
  auto log = spdlog::get(logName);

  if (log != nullptr) {
    log->critical("Logger was redefined {}", logName);
  } else {
    int logLevel = m_env_log_level_all;
    LoggerPtr newLogger = define_logger(logName);

    char* env_log_level_type =
        getenv((std::string("PT_HPU_LOG_LEVEL_") + logName).c_str());

    if (env_log_level_type != nullptr) {
      logLevel = std::stoi(env_log_level_type);
    } else {
      unsigned long logMask = getLogMask(logType);
      if (logMask == NOT_SUPPORTED_LOG_MASK) {
        logLevel = 5;
      } else if ((logMask & m_env_log_module_mask) == logMask) {
        logLevel = m_env_log_type;
      }
    }
    m_loggersMap[(uint32_t)logType] = newLogger;
    newLogger->set_pattern(m_logPattern);
    set_log_level(logType, logLevel);
  }
}

void LogManager::drop_log(const LogManager::LogType& logType) {
  getLogger(logType, "drop_log")->flush();
  m_loggersMap[(uint32_t)logType].reset();
  m_loggerLevels[(uint32_t)logType] = spdlog::level::off;
  spdlog::drop(getLogTypeString(logType));
}

void LogManager::set_log_level(
    const LogManager::LogType& logType,
    unsigned log_level) {
  assert(log_level >= spdlog::level::trace && log_level <= spdlog::level::off);

  LoggerPtr log = getLogger(logType, "set log level");
  log->set_level(spdlog::level::level_enum(log_level));
  m_loggerLevels[(uint32_t)logType] = log_level;
}

unsigned LogManager::get_log_level(const LogManager::LogType& logType) {
  if (unlikely(
          logType >= LogType::LOG_MAX || !m_loggersMap[(uint32_t)logType])) {
    return validate(getLogTypeString(logType), "get_log_level")->level();
  }
  return m_loggerLevels[(uint32_t)logType];
}

void* LogManager::enableConsole(const LogManager::LogType& logType) {
  std::vector<spdlog::sink_ptr>& sinks =
      getLogger(logType, "set sink level")->sinks();
  auto console = std::make_shared<COLOR_MACRO>();
  sinks.push_back(console);
  return static_cast<void*>(console.get());
}

bool LogManager::disableConsole(
    const LogManager::LogType& logType,
    void* console) {
  bool ret = false;
  std::vector<spdlog::sink_ptr>& sinks =
      getLogger(logType, "set sink level")->sinks();
  auto it = sinks.begin();
  for (; it != sinks.end(); it++) {
    if (it->get() == console)
      break;
  }
  if (it != sinks.end()) {
    sinks.erase(it);
    ret = true;
  }
  return ret;
}

void LogManager::set_logger_sink(
    const LogManager::LogType& logType,
    const std::string& pathname,
    unsigned lvl,
    size_t size,
    size_t amount) {
  const char* enable_file_colors = getenv("ENABLE_LOG_FILE_COLORS");
  bool should_do_colors =
      enable_file_colors && !strcmp(enable_file_colors, "true");
  spdlog::sink_ptr sink =
      std::make_shared<spdlog::sinks::ansicolor_rotating_file_sink_mt>(
          pathname, size, amount, should_do_colors);
  std::vector<spdlog::sink_ptr>& sinks =
      getLogger(logType, "set sink level")->sinks();
  switch (lvl) {
    case spdlog::level::trace:
      sink->set_level(spdlog::level::trace);
      break;
    case spdlog::level::debug:
      sink->set_level(spdlog::level::debug);
      break;
    case spdlog::level::info:
      sink->set_level(spdlog::level::info);
      break;
    case spdlog::level::warn:
      sink->set_level(spdlog::level::warn);
      break;
    case spdlog::level::err:
      sink->set_level(spdlog::level::err);
      break;
    case spdlog::level::critical:
      sink->set_level(spdlog::level::critical);
      break;
    case spdlog::level::off:
      return;
    default:
      assert(0 && "No such log level");
  };
  sinks.push_back(sink);
}

void LogManager::clearLogContext() {
  if (s_logContextStack.size()) {
    s_logContext = s_logContextStack.back();
    s_logContextStack.pop_back();
  } else {
    s_logContext.clear();
  }
}

LogManager::LoggerPtr LogManager::getLogger(
    const LogManager::LogType& logType,
    const std::string& msg) const {
  auto loggerPtr = m_loggersMap[(uint32_t)logType];
  if (unlikely(logType >= LogType::LOG_MAX || !loggerPtr)) {
    return validate(getLogTypeString(logType), msg);
  }
  return loggerPtr;
}

void LogManager::log_wrapper(
    const LogManager::LogType& logType,
    const int logLevel,
    std::string&& s) {
  LoggerPtr pLog = getLogger(logType, s);
  spdlog::level::level_enum level =
      static_cast<spdlog::level::level_enum>(logLevel);

  if (!pLog->should_log(level)) {
    return;
  }

  if (g_traceModeLogging) {
    level = spdlog::level::trace;
  }
  std::unique_lock<std::mutex> mlock(m_mutex);
  if (s_threadIdx == -1) {
    s_threadIdx = s_numThreads++;
  }
  pLog->log(level, s_logContext + s);
}

void LogManager::setLogPattern() {
  m_printFileAndLine = false; /* TODO: add glabal var */
  m_logPattern = "";
  m_logPattern += "[%T.%f]";
  m_logPattern +=
      "[%-" + std::to_string(LOGGER_NAME_MAX_LENGTH) + "n][%^%-5l%$] %v";
  spdlog::set_pattern(m_logPattern);
}

const std::string& LogManager::getLogTypeString(
    const LogManager::LogType& logType) const {
  static const std::string s_logTypeString[] = {
      "DEVICE",
      "KERNEL",
      "BRIDGE",
      "SYNHELPER",
      "DISTRIBUTED",
      "LAZY",
      "HABANAHOOKS",
      "FALLBACK",
      "STATS",
      "TEST",
      "DYNAMIC_SHAPE",
      "DEVMEM",
      "HABHELPER"};
  COMPILE_TIME_ASSERT_VERIFY_ARRAY_SIZE(
      s_logTypeString, (uint32_t)LogType::LOG_MAX);
  return s_logTypeString[(uint32_t)logType];
}

unsigned long LogManager::getLogMask(const LogManager::LogType& logType) const {
  switch (logType) {
    case LogType::DEVICE:
      return DEVICE_LOG_MASK;
    case LogType::KERNEL:
      return KERNEL_LOG_MASK;
    case LogType::BRIDGE:
      return BRIDGE_LOG_MASK;
    case LogType::SYNHELPER:
      return SYNHELPER_LOG_MASK;
    case LogType::DISTRIBUTED:
      return DISTRIBUTED_LOG_MASK;
    case LogType::LAZY:
      return LAZY_LOG_MASK;
    case LogType::HABANAHOOKS:
      return HABANAHOOKS_LOG_MASK;
    case LogType::FALLBACK:
      return FALLBACK_LOG_MASK;
    case LogType::STATS:
      return STATS_LOG_MASK;
    case LogType::TEST:
      return TEST_LOG_MASK;
    case LogType::DYNAMIC_SHAPE:
      return DYNAMIC_SHAPE_LOG_MASK;
    case LogType::DEVMEM:
      return DEVMEM_LOG_MASK;
    case LogType::HABHELPER:
      return HABHELPER_LOG_MASK;
    default:
      assert(0 && "LogManager::getLogMask: Unsuported log type");
      return NOT_SUPPORTED_LOG_MASK;
  }
}

} // namespace ptspdlogger
