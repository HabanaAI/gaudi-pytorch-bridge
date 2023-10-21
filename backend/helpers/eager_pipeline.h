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
#include <absl/base/thread_annotations.h>
#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"
#include "pytorch_helpers/habana_helpers/thread_queue.h"

namespace habana_helpers {

/**
 * Controls underlying thread pool execution.
 * Thread safe.
 */
class ThreadPoolControl {
 public:
  /**
   * @param num_threads Number of available threads in underlying pool
   */
  explicit ThreadPoolControl(){};

  ThreadPoolControl(const ThreadPoolControl&) = delete;
  ThreadPoolControl& operator=(const ThreadPoolControl&) = delete;
  ThreadPoolControl(ThreadPoolControl&&) = delete;
  ThreadPoolControl& operator=(ThreadPoolControl&&) = delete;
  ~ThreadPoolControl() = default;

  /**
   * Wait until thread pool is done.
   * Thread safe.
   */
  void JoinPendingThread() {
    std::shared_future<void> shared_thread_handle;
    {
      std::unique_lock lock{m_thread_handle_mutex};
      shared_thread_handle = m_thread_handle;
    }
    if (shared_thread_handle.valid()) {
      PT_LAZY_EXEC_THREAD("Waiting for thread to finish");
      try {
        shared_thread_handle.get();
      } catch (std::exception&) {
      }
    }
  }

  /**
   * Schedules work to thread pool and stores handle to scheduled work.
   * Thread safe.
   *
   * @param f Function with work
   * @param args Arguments to work
   */
  template <class F, class... Args>
  void ScheduleWorkAndUpdateThreadHandle(F&& f, Args&&... args) {
    auto handle = m_thread_pool_obj.enqueue<F, Args...>(
        std::forward<F>(f), std::forward<Args>(args)...);
    std::unique_lock lock{m_thread_handle_mutex};
    m_thread_handle = std::move(handle);
  }

 private:
  /**
   * Underlying thread pool.
   */
  habana_helpers::ThreadPool m_thread_pool_obj;

  /**
   * Handle to last scheduled work in thread pool.
   */
  std::shared_future<void> m_thread_handle GUARDED_BY(m_thread_handle_mutex);

  /**
   * Guarding accesses to thread handle.
   */
  std::mutex m_thread_handle_mutex;

  /**
   * Busy waits until underlying thread pool is stopped or still has items.
   */
};

/**
 * Exposed thread pool that is used for graph compilation.
 */
class Singleton_CompileThreadPool {
 public:
  /**
   * Returns reference to ThreadPoolControl that is controlling graph
   * compilation thread pool.
   */
  static ThreadPoolControl& getInstance() {
    static ThreadPoolControl thread_pool_control_obj;
    return thread_pool_control_obj;
  }

 private:
  static constexpr size_t num_threads = 1;
  Singleton_CompileThreadPool() = default;
  Singleton_CompileThreadPool(const Singleton_CompileThreadPool&) = delete;
  Singleton_CompileThreadPool& operator=(const Singleton_CompileThreadPool&) =
      delete;
};

/**
 * Exposes thread pool that is used for graph execution.
 */
class Singleton_ExecThreadPool {
 public:
  /**
   * Returns reference to ThreadPoolControl that is controlling graph execution
   * thread pool.
   */
  static ThreadPoolControl& getInstance() {
    static ThreadPoolControl thread_pool_control_obj;
    return thread_pool_control_obj;
  }

 private:
  Singleton_ExecThreadPool() = default;
  Singleton_ExecThreadPool(const Singleton_ExecThreadPool&) = delete;
  Singleton_ExecThreadPool& operator=(const Singleton_ExecThreadPool&) = delete;
};

} // namespace habana_helpers
