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
#include <sys/types.h>
#include <unistd.h>
#include "habana_lazy/debug_utils.h"

namespace Logger {

uint64_t get_tid_internal() {
  static thread_local uint64_t tid{static_cast<uint64_t>(syscall(__NR_gettid))};
  return tid;
}

uint64_t get_rank_internal() {
  uint64_t node_id = 0;
  char* node_id_ptr = std::getenv("ID");
  if (node_id_ptr != nullptr) {
    node_id = std::stoul(node_id_ptr, nullptr, 16);
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
          line,
          " (most recent call first):\n",
          c10::get_backtrace(1)));
}
} // namespace Logger
