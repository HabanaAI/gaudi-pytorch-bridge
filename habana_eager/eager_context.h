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
#pragma once

#include <future>
#include "backend/helpers/eager_pipeline.h"

namespace habana {
namespace eager {

/**
 * Class to store the eager context which we might need across the Ops.
 * This might have members/functionalities related to pipelining as well
 * other eager development/feature.
 */
class SingleTonEagerContext {
 public:
  /**
   * Obtains instance of context.
   * Thread safe.
   */
  static SingleTonEagerContext& getInstance() {
    static SingleTonEagerContext eager_context_obj;
    return eager_context_obj;
  }

  /**
   * Schedule work and update handle to last scheduled work.
   * Thread safe.
   *
   * @param starter Function that launch execution and returns handle
   */
  void ScheduleWorkAndUpdateLoweringThreadHandle(
      const std::function<std::shared_future<void>()>& starter);

  /**
   * Joins scheduled work.
   * Ensures handle is properly obtained without data races, thus thread safe.
   */
  void JoinPendingLoweringThread();

  /**
   * Saves exception that happened in lowering task.
   * Not thread safe.
   */
  void StoreLoweringThreadException(std::exception_ptr exception) {
    m_lowering_thread_exception = std::move(exception);
  }

  /**
   * Processes stored exception, and fatal error main thread in case it was
   * present. Not thread safe.
   */
  void HandleException();

 private:
  SingleTonEagerContext() = default;
  SingleTonEagerContext(const SingleTonEagerContext&) = delete;
  SingleTonEagerContext& operator=(const SingleTonEagerContext&) = delete;

  /**
   * Saved exception.
   */
  std::exception_ptr m_lowering_thread_exception = nullptr;

  /**
   * Handle to last scheduled task (assumption - only 1 task can execute in
   * parallel, thus FIFO order of completion of scheduled work is maintained)
   */
  std::shared_future<void> m_lowering_thread_handle;

  /**
   * Mutex to ensure thread safety of storing and restoring task handles.
   */
  std::mutex m_lowering_thread_handle_mutex;
};

void JoinPendingPipelineThreads();

} // namespace eager
} // namespace habana
