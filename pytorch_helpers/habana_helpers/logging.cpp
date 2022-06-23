/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "logging.h"
#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include "habana_lazy/debug_utils.h"

namespace Logger {
void habana_assert(
    const char* func,
    const char* file,
    uint32_t line,
    const std::string& msg) {
  throw c10::Error(
      msg,
      Logger::str(
          "Habana exception raised from ",
          func,
          " at ",
          c10::detail::StripBasename(file),
          ":",
          line,
          " (most recent call first):\n",
          c10::get_backtrace(1)));
}
} // namespace Logger


