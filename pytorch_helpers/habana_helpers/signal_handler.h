/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <signal.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

/**
 * The signal handler is modelled after caffe2/utils/signal_handler.cc from
 * pytorch frameworks, which is by default not compiled in the framework.
 */
namespace habana_helpers {
namespace signalHandler {

using signalHandlerFnPtr =
    std::add_pointer<void(int signum, siginfo_t* info, void* ctx)>::type;

// This is list of all the fatal signals we catch here
struct {
  const char* name;
  int signum;
  struct sigaction previous;
} SignalHandlersList[] = {
    {"SIGABRT", SIGABRT, {}},
    {"SIGINT", SIGINT, {}},
    {"SIGILL", SIGILL, {}},
    {"SIGFPE", SIGFPE, {}},
    {"SIGBUS", SIGBUS, {}},
    {"SIGSEGV", SIGSEGV, {}},
    {nullptr, 0, {}}};

signalHandlerFnPtr registeredHandler;

struct sigaction* GetPreviousSigaction(int signum);

const char* GetSignalName(int signum);

void CallPreviousSignalHandler(
    struct sigaction* action,
    int signum,
    siginfo_t* info,
    void* ctx);

void HabanaSignalHandler(int signum, siginfo_t* info, void* ctx);

// This function will be called to install the signal handler
// for handling fatal signals
void InstallSignalHandlers(signalHandlerFnPtr handlerFn);

void fatalSignalHandler(int signum, siginfo_t* info, void* ctx);
} // namespace signalHandler
} // namespace habana_helpers
