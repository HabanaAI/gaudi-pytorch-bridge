/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "habana_helpers/logging.h"
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include "habana_helpers/pt_version_check.h"
#include <c10/util/Lazy.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/debug_utils.h"

// -------------- HL LOG ----------------
namespace HlLogger {
// create loggers (all the log files are created immediately when the module is
// loaded)
static void createModuleLoggers(LoggerType) {}

// all the following functions are optional and any/all of them can be omitted

static void createModuleLoggerOnDemandForTowl() {
  hl_logger::LoggerCreateParams default_params;

  if (GET_ENV_FLAG_NEW(PT_TOWL_LOG_SEPARATED_FILE)) {
    default_params.logFileName = "towl_log.txt";
  } else {
    default_params.logFileName = "pytorch_log.txt";
  }
  default_params.rotateLogfileOnOpen = true;
  default_params.logFileAmount = GET_ENV_FLAG_NEW(PT_TOWL_LOG_FILE_AMOUNT);
  default_params.logFileSize = 3u * 1024u * 1024ul * 1024u;
  default_params.logFileBufferSize = 4u * 1024u * 1024u;
  default_params.defaultLoggingLevel = HLLOG_LEVEL_DEBUG;
  default_params.forceDefaultLoggingLevel = true;
  hl_logger::createLoggersOnDemand({LoggerType::PT_TOWL}, default_params);
}
// on-demand loggers
// log files created when the first message is logged into such logger
// this is a recommended way of loggers creation
static void createModuleLoggersOnDemand(LoggerType) {
  hl_logger::LoggerCreateParams default_params, trace_params;
  default_params.logFileName = "pytorch_log.txt";
  default_params.logFileAmount = GET_ENV_FLAG_NEW(PT_LOG_FILE_AMOUNT);
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
       LoggerType::PT_CUSTOM,      LoggerType::PT_RECIPE_STATS,
       LoggerType::PT_HPUGRAPH,    LoggerType::PT_CONST_SECTION,
       LoggerType::PT_PYTHON},
      default_params);

  trace_params.logFileName = "pytorch_log.txt";
  trace_params.defaultLoggingLevel = HLLOG_LEVEL_TRACE;
  trace_params.forceDefaultLoggingLevel = true;

  hl_logger::createLoggerOnDemand(LoggerType::PT_TRACE, trace_params);
  // Guarded by additional flag to not enable towl logger
  // by using common flags like LOG_LEVEL_ALL_PT
  if (true or GET_ENV_FLAG_NEW(PT_TOWL_LOG_ENABLE)) {
    createModuleLoggerOnDemandForTowl();
  }
}

// a callback when a dtor of your module is called (e.g. close an app, dlclose,
// etc) usually is used to log a final message
static void onModuleLoggersBeforeDestroy(LoggerType) {
  HLLOG_INFO(
      PT_BRIDGE,
      "Closing PyTorch logger. No more log messages will be logged.");
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
    PT_RECIPE_STATS,
    PT_HPUGRAPH,
    PT_CONST_SECTION,
    PT_PYTHON,
    PT_TOWL,
    LOG_MAX)
// -------------- HL LOG ----------------

namespace Logger {

std::string synStatusToStr(synStatus statusArg) {
  static std::mutex mtx{};
  static std::vector<std::string> statusStr(
      static_cast<int>(synStatus::synStatusLast), std::string(""));

  auto idx = static_cast<size_t>(statusArg);
  std::unique_lock<std::mutex> lock(mtx);
  if (statusStr[idx].empty()) {
    char statusDescription[STATUS_DESCRIPTION_MAX_SIZE];
    auto isDescriptionValid =
        synStatusGetBriefDescription(
            statusArg, statusDescription, STATUS_DESCRIPTION_MAX_SIZE) ==
        synStatus::synSuccess;

    if (isDescriptionValid) {
      statusStr[idx] = std::string(statusDescription);
      return statusStr[idx];
    } else {
      PT_BRIDGE_WARN("Could not get translation for synStatus: ", statusArg);
      return std::string("UnkownDescription");
    }
  }

  return statusStr[idx];
}

std::string formatStatusMsg(synStatus statusArg) {
  return fmt::format(
      "synStatus {} [{}]. ", statusArg, synStatusToStr(statusArg));
}

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
  std::string logmsg = Logger::str(
      "[Rank:",
      Logger::get_rank(),
      "] ",
      "Habana exception raised from ",
      func,
      " at ",
      c10::detail::StripBasename(file),
      ":",
      line);
  typedef std::shared_ptr<c10::PrecomputedLazyValue<std::string>> MsgPtr;
  MsgPtr msgPtr(new c10::PrecomputedLazyValue<std::string>(logmsg));
  throw c10::Error(msg, msgPtr);
}

} // namespace Logger
