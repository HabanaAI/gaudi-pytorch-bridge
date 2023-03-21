/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_helpers/logging.h"
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "habana_lazy/debug_utils.h"

// -------------- HL LOG ----------------
namespace HlLogger {
// create loggers (all the log files are created immediately when the module is
// loaded)
static void createModuleLoggers(LoggerType) {}

// all the following functions are optional and any/all of them can be omitted

// on-demand loggers
// log files created when the first message is logged into such logger
// this is a recommended way of loggers creation
static void createModuleLoggersOnDemand(LoggerType) {
  hl_logger::LoggerCreateParams default_params, trace_params;
  default_params.logFileName = "pytorch_log.txt";
  hl_logger::createLoggersOnDemand(
      {LoggerType::PT_DEVICE,      LoggerType::PT_KERNEL,
       LoggerType::PT_BRIDGE,      LoggerType::PT_SYNHELPER,
       LoggerType::PT_DISTRIBUTED, LoggerType::PT_LAZY,
       LoggerType::PT_FALLBACK,    LoggerType::PT_STATS,
       LoggerType::PT_TEST,        LoggerType::PT_DYNAMIC_SHAPE,
       LoggerType::PT_DEVMEM,      LoggerType::PT_HABHELPER,
       LoggerType::PT_IRGRAPH,     LoggerType::PT_VIEWTABLE,
       LoggerType::PT_REFINEMENT,  LoggerType::PT_HOSTSTAT,
       LoggerType::PT_LAYOUTS,     LoggerType::PT_PARALLEL_ACC,
       LoggerType::PT_LAZY_EAGER,  LoggerType::PT_MEMLOG,
       LoggerType::PT_EXEC_THREAD, LoggerType::PT_EAGER,
       LoggerType::PT_CUSTOM},
      default_params);

  trace_params.logFileName = "pytorch_log.txt";
  trace_params.defaultLoggingLevel = HLLOG_LEVEL_TRACE;
  trace_params.forceDefaultLoggingLevel = true;
  hl_logger::createLoggerOnDemand(LoggerType::PT_TRACE, trace_params);
}

// a callback when a dtor of your module is called (e.g. close an app, dlclose,
// etc) usually is used to log a final message
static void onModuleLoggersBeforeDestroy(LoggerType) {
  HLLOG_INFO(
      PT_BRIDGE,
      "Closing PyTorch logger. No more log messages will be logged.");
}

// a callback when an app got a signal (usually it means a crash)
// can be used to log a stacktrace or any other info
static void onModuleLoggersCrashSignal(
    LoggerType,
    int signal,
    const char* signalStr,
    bool isSevere) {
  HLLOG_ERR(
      PT_BRIDGE,
      "Crash. signal : {} {}. Severity: {}",
      signal,
      signalStr,
      isSevere ? "high" : "low");
  hl_logger::logStacktrace(
      LoggerType::PT_BRIDGE, isSevere ? HLLOG_LEVEL_ERROR : HLLOG_LEVEL_INFO);
}

} // namespace HlLogger

// define logger internal variables. requires a list of all the logger names
// (for string representation)
HLLOG_DEFINE_MODULE_LOGGER(
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
    PT_CUSTOM,
    LOG_MAX)
// -------------- HL LOG ----------------

namespace Logger {

uint64_t get_tid_internal() {
  static thread_local uint64_t tid{static_cast<uint64_t>(syscall(__NR_gettid))};
  return tid;
}

uint64_t get_rank_internal() {
  uint64_t node_id = 0;
  char* node_id_ptr = std::getenv("RANK");
  if (node_id_ptr != nullptr) {
    node_id = std::stoul(node_id_ptr, nullptr, 10);
  }
  return node_id;
}

void habana_assert(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg) {
  throw c10::Error(
      msg,
      Logger::str(
          Logger::print_hdr(),
          "Habana exception raised from ",
          func,
          " at ",
          c10::detail::StripBasename(file),
          ":",
          line));
}

} // namespace Logger
