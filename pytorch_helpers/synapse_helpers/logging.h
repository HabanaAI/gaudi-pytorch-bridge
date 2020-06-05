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

#define HABANA_ASSERT(condition)                                                                  \
  {                                                                                               \
    if (!(condition)) {                                                                           \
      LOG_(FATAL) << "Assertion (" << #condition << ") is false! " << __FILE__ << ":" << __LINE__; \
    }                                                                                             \
  }

#ifdef GENERIC_HELPERS
// TODO: use glog
#include <iostream>

class VerboseLogger {
 public:
  VerboseLogger(std::ostream& out, unsigned level) : out_{out}, level_{level} {}
  ~VerboseLogger() { out_ << "\n"; }

  template <typename T>
  friend std::ostream& operator<<(VerboseLogger&& log, T&& t) {
    return log.out_ << std::forward<T>(t);
  }

 private:
  std::ostream& out_;
  unsigned level_;
};

class PotentiallyFatalLogger {
 public:
  PotentiallyFatalLogger(std::ostream& out, std::string level) : out_{out}, level_{std::move(level)} {}
  ~PotentiallyFatalLogger() {
    out_ << "\n";
    if (level_ == "FATAL") {
      std::terminate();
    }
  }

  template <typename T>
  friend std::ostream& operator<<(PotentiallyFatalLogger&& log, T&& t) {
    return log.out_ << std::forward<T>(t);
  }

 private:
  std::ostream& out_;
  std::string level_;
};

#define LOG_(LEVEL) PotentiallyFatalLogger(std::clog, #LEVEL)
#define VLOG_(LEVEL) VerboseLogger(std::clog, (LEVEL))
#else
#include <tensorflow/core/platform/default/logging.h>
#endif
