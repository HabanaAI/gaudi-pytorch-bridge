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

#include <iostream>
#include "pytorch_helpers/habana_helpers/logging.h"

class VerboseLogger {
 public:
  VerboseLogger(std::ostream& out, unsigned level) : out_{out}, level_{level} {}
  ~VerboseLogger() {
    out_ << "\n";
  }

  template <typename T>
  friend std::ostream& operator<<(VerboseLogger&& log, T&& t) {
    if (PtLogger::getLogger()->getModuleMask() &
        (PtLogger::ModuleMask::SYNHELPER)) {
      return log.out_ << "Deprecated! " << std::forward<T>(t);
    } else {
      return log.out_;
    }
  }

 private:
  std::ostream& out_;
  unsigned level_;
};

class PotentiallyFatalLogger {
 public:
  PotentiallyFatalLogger(std::ostream& out, std::string level)
      : out_{out}, level_{std::move(level)} {}
  ~PotentiallyFatalLogger() {
    out_ << "Deprecated! "
         << "\n";
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

// Safer to use cerr which is tied to cout for warnings and fatal errors as it
// is unbuffered
#define LOG_(LEVEL) PotentiallyFatalLogger(std::cerr, #LEVEL)
// clog is buffered and is might be more efficient for verbose logging
#define VLOG_(LEVEL) VerboseLogger(std::clog, (LEVEL))

#define HABANA_ASSERT(condition)                                               \
  {                                                                            \
    if (!(condition)) {                                                        \
      LOG_(FATAL) << "Assertion (" << #condition << ") is false! " << __FILE__ \
                  << ":" << __LINE__;                                          \
    }                                                                          \
  }
