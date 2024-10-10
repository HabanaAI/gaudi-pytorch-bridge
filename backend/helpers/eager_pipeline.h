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
#include "pytorch_helpers/habana_helpers/python_utils.h"
#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"
#include "pytorch_helpers/habana_helpers/thread_queue.h"

namespace habana_helpers {

/**
 * Controls underlying thread pool execution.
 * Thread safe.
 */
class ThreadPoolControl {
 public:
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
    try {
      habana_helpers::AutoNoGIL gil_release;
      m_thread_pool_obj.waitWorkComplete();
    } catch (const std::exception& e) {
      PT_BRIDGE_WARN("Exception caught in thread...\n", e.what());
      throw;
    } catch (...) {
      PT_BRIDGE_WARN("Exception caught in thread...\n");
      throw;
    }
  }

  /**
   * Schedules work to thread pool
   *
   * @param f Function with work
   * @param args Arguments to work
   */
  template <class F, class... Args>
  void Enqueue(F&& f, Args&&... args) {
    m_thread_pool_obj.enqueue<F, Args...>(
        std::forward<F>(f), std::forward<Args>(args)...);
  }

 private:
  /**
   * Underlying thread pool.
   */
  habana_helpers::ThreadPool m_thread_pool_obj{true};
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

/**
 * Exposes thread pool that is used for garbage collection.
 */
class Singleton_GarbageCollectionThreadPool {
 public:
  /**
   * Returns reference to ThreadPoolControl that is controlling garbage
   * collector thread pool.
   */
  static ThreadPoolControl& getInstance() {
    static ThreadPoolControl thread_pool_control_obj;
    return thread_pool_control_obj;
  }

 private:
  Singleton_GarbageCollectionThreadPool() = default;
  Singleton_GarbageCollectionThreadPool(
      const Singleton_GarbageCollectionThreadPool&) = delete;
  Singleton_GarbageCollectionThreadPool& operator=(
      const Singleton_GarbageCollectionThreadPool&) = delete;
};

} // namespace habana_helpers
