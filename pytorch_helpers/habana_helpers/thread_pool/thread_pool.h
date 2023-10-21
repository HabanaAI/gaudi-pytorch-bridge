/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include <unistd.h>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "backend/synapse_helpers/env_flags.h"
#include "pytorch_helpers/habana_helpers/thread_queue.h"

namespace habana_helpers {

// add new work item to the pool
template <typename T>
class BlockingQueue {
 public:
  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }
  void push(T&& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    lock.unlock();
    cond_.notify_one();
  }
  bool empty() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
};

template <template <typename> typename Queue>
class ThreadPoolBase {
 public:
  ThreadPoolBase(bool propagate_exception = false);
  ~ThreadPoolBase();

  using Task = std::packaged_task<void()>;

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args);
  void waitWorkComplete();
  void rethrowIfException();
  std::string ToString() const;

 private:
  Queue<Task> tasks_;

  std::thread thread_;
  std::atomic_bool stop_;
  std::exception_ptr ex_ptr_;

  bool propagate_exception_ = false;

  void main_loop() {
    while (!stop_)
      executePendingTask(std::move(tasks_.pop()));
  }
  void executePendingTask(Task&& task);
};

template <template <typename> typename Queue>
template <class F, class... Args>
auto ThreadPoolBase<Queue>::enqueue(F&& f, Args&&... args) {
  auto packed_func = [args = std::make_tuple(std::forward<Args>(args)...),
                      func = std::move(f)]() mutable {
    std::apply([&](auto&&... x) { func(std::forward<Args>(x)...); }, args);
  };
  auto task = std::packaged_task<void()>(std::move(packed_func));
  auto res = task.get_future();
  tasks_.push(std::move(task));
  return res;
};

using ThreadPool = ThreadPoolBase<BlockingQueue>;
} // namespace habana_helpers
