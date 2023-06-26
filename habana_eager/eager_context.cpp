/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "eager_context.h"
#include <c10/macros/Macros.h>
#include <future>
#include <memory>
#include <mutex>
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/python_utils.h"

namespace habana {
namespace eager {

void SingleTonEagerContext::JoinPendingLoweringThread() {
  PT_EAGER_TRACE;

  std::shared_future<void> shared_lowering_thread_handle;
  {
    std::unique_lock lock{m_lowering_thread_handle_mutex};
    shared_lowering_thread_handle = m_lowering_thread_handle;
  }
  if (shared_lowering_thread_handle.valid()) {
    PT_LAZY_EXEC_THREAD("Waiting for lowering thread to finish");

    habana_helpers::AutoNoGIL gil_release;

    // If the future is already ready when below line executes, it can
    // create an exception. Ignore the exception as the wait is already
    // over.
    try {
      shared_lowering_thread_handle.get();
    } catch (...) {
    };
  }
  HandleException();
}

void SingleTonEagerContext::ScheduleWorkAndUpdateLoweringThreadHandle(
    const std::function<std::shared_future<void>()>& starter) {
  std::unique_lock lock{m_lowering_thread_handle_mutex};
  m_lowering_thread_handle = starter();
}

void SingleTonEagerContext::HandleException() {
  PT_EAGER_TRACE;
  if (C10_UNLIKELY(m_lowering_thread_exception)) {
    try {
      std::rethrow_exception(m_lowering_thread_exception);
    } catch (const std::exception& e) {
      m_lowering_thread_exception = nullptr;
      PT_BRIDGE_FATAL("Exception in Lowering thread...\n", e.what());
    } catch (...) {
      m_lowering_thread_exception = nullptr;
      PT_BRIDGE_FATAL("Exception in Lowering thread...\n");
    }
  }
}

} // namespace eager
} // namespace habana
