/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include "habana_helpers/signal_handler.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/env_flags.h"

#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <mutex>

/**
 * The signal handler is modelled after caffe2/utils/signal_handler.cc from
 * pytorch frameworks, which is by default not compiled in the framework.
 */
namespace habana_helpers {
namespace signalHandler {

struct sigaction* GetPreviousSigaction(int signum) {
  for (auto handler = SignalHandlersList; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return &handler->previous;
    }
  }
  return nullptr;
}

const char* GetSignalName(int signum) {
  for (auto handler = SignalHandlersList; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return handler->name;
    }
  }
  return nullptr;
}

void CallPreviousSignalHandler(
    struct sigaction* action,
    int signum,
    siginfo_t* info,
    void* ctx) {
  if (!action->sa_handler) {
    return;
  }
  if ((action->sa_flags & SA_SIGINFO) == SA_SIGINFO) {
    action->sa_sigaction(signum, info, ctx);
  } else {
    action->sa_handler(signum);
  }
}

void HabanaSignalHandler(int signum, siginfo_t* info, void* ctx) {
  // This should be in the SignalHandler list
  const char* name = GetSignalName(signum);

  // If there is no signal registered, we should just return? How does
  // it reach here anyway?
  if (!name) {
    CallPreviousSignalHandler(GetPreviousSigaction(signum), signum, info, ctx);
    return;
  }

  // Call the registered handler function, with the siginfo and ctx
  registeredHandler(signum, info, ctx);

  // Raise the signal again in case of a previous signal handler
  if (GetPreviousSigaction(signum)) {
    sigaction(signum, GetPreviousSigaction(signum), nullptr);
    raise(signum);
  }
}

// This function will be called to install the signal handler
// for handling fatal signals
void InstallSignalHandlers(signalHandlerFnPtr handlerFn) {
  // Install only once
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    struct sigaction sa;
    // Mask all signals during the handler
    sigfillset(&sa.sa_mask);
    // Use alternate stack for signal handler and also pass siginfo
    sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
    // Install the signal handler
    sa.sa_sigaction = HabanaSignalHandler;
    registeredHandler = handlerFn;

    // Do a sigaction for all fatal signals we want to handle, with the
    // same action.
    for (auto* handler = SignalHandlersList; handler->name != nullptr;
         handler++) {
      if (sigaction(handler->signum, &sa, &handler->previous)) {
        std::string str("Failed to add ");
        str += handler->name;
        str += " handler!";
        perror(str.c_str());
      }
    }
  });
}

class HPUSigHandler {
 public:
  HPUSigHandler() {
    if (GET_ENV_FLAG_NEW(PT_HPU_ERROR_HANDLER)) {
      habana_helpers::signalHandler::InstallSignalHandlers(fatalSignalHandler);
      done = false;
    }
  }
  bool CmpExcgDone() {
    bool expected = false;
    bool desired = true;
    return done.compare_exchange_strong(expected, desired);
  }

 private:
  std::atomic<bool> done;
};

static HPUSigHandler sigHandlerStaticInstance;

// Fatal signal handler for Habana
void fatalSignalHandler(int signum, siginfo_t* info, void* ctx) {
  static_cast<void>(info); // Avoid warnings for unused arguments
  static_cast<void>(ctx); // Avoid warnings for unused arguments

  // Atmocally set the flag to true so that even if two thread raise
  // signal simultaneously and reach here at the same time, there will
  // be race condition on the done flag and only one thread will perform
  // the cleanup.
  if (sigHandlerStaticInstance.CmpExcgDone()) {
    std::stringstream ss;
    if (signum == SIGINT) {
      ss << "Received " << strsignal(signum) << "\n";
    } else {
      ss << "Internal Error: Received signal - " << strsignal(signum) << "\n";
    }

    if (GET_ENV_FLAG_NEW(PT_HPU_PRINT_BACKTRACE_ON_SIGNAL)) {
      ss << c10::get_backtrace(2) << "\n";
    }

    std::cerr << ss.str();

    // Exit immediately
    if ((signum == SIGINT) || (signum == SIGTERM)) {
      _exit(signum);
    }
  };
}

} // namespace signalHandler
} // namespace habana_helpers
