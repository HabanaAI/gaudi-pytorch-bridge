/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace habana_lazy {

class AccThreadPool {
 public:
  using AccTask = std::function<void()>;

  AccThreadPool();
  ~AccThreadPool();

  void run(std::function<void()>&& func);
  void waitWorkComplete();
  void discardPendingTasks();
  bool inAccThreadContext() const;

 private:
  bool inThreadPool() const;
  std::queue<AccTask> tasks_;
  std::thread thread_;
  mutable std::mutex mutex_;
  std::atomic_bool running_;
  std::atomic_bool stop_;
  static thread_local bool task_in_progress_;
  std::atomic<std::size_t> task_count_;
  std::exception_ptr ex_ptr_;

  // Check if no exception has been thrown by any task_. If excpetion occured
  // then rethrow it in the main thread.
  void checkNoException();
  void main_loop();
  void executePendingTask();
};

} // namespace habana_lazy