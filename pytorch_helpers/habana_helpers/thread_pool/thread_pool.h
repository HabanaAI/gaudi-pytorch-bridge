/**
 * Copyright (c) 2021-2024 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include <type_traits>
#include <vector>

#include "backend/synapse_helpers/env_flags.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana_helpers {

// Should be replaced by C++23 std::move_only_function
class move_only_function_void {
 public:
  template <typename F>
  move_only_function_void(F&& f)
      : func_wrapper_(
            std::make_unique<Wrapper<std::decay_t<F>>>(std::forward<F>(f))) {}

  void operator()() {
    func_wrapper_->invoke();
  }

 private:
  struct BaseWrapper {
    virtual void invoke() = 0;
    virtual ~BaseWrapper() = default;
  };

  template <typename F>
  struct Wrapper : public BaseWrapper {
    F func_;
    template <typename T>
    Wrapper(T&& f) : func_(std::forward<T>(f)) {}
    void invoke() override {
      func_();
    }
  };
  std::unique_ptr<BaseWrapper> func_wrapper_;
};

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

struct SingleThreadPolicy {
  static constexpr uint64_t GetNumThreads() {
    return 1;
  }
};

struct MultiThreadPolicy {
  static uint64_t GetNumThreads();
};

template <
    template <typename>
    typename Queue,
    typename Task,
    typename ThreadPolicy>
class ThreadPoolBase {
 public:
  ThreadPoolBase(
      bool propagate_exception = false,
      const std::function<void()>& init_thread = nullptr);

  template <
      typename T = ThreadPolicy,
      typename std::enable_if_t<std::is_same_v<T, MultiThreadPolicy>, bool> =
          true>
  ThreadPoolBase(
      bool propagate_exception,
      const std::function<void()>& init_thread,
      uint64_t threads_number);

  ~ThreadPoolBase();

  template <
      typename T = Task,
      typename std::
          enable_if_t<std::is_same_v<T, move_only_function_void>, bool> = true>
  void enqueue(T&& task);

  template <
      class F,
      class... Args,
      typename T = Task,
      typename std::
          enable_if_t<std::is_same_v<T, move_only_function_void>, bool> = true>
  void enqueue(F&& f, Args&&... args);

  template <
      typename T = Task,
      typename std::
          enable_if_t<std::is_same_v<T, move_only_function_void>, bool> = true>
  void waitWorkComplete();

  template <
      class F,
      class... Args,
      typename T = Task,
      typename std::enable_if_t<
          std::is_same_v<T, std::packaged_task<void()>>,
          bool> = true>
  std::future<void> enqueue(F&& f, Args&&... args);

  void RethrowIfException();
  std::string ToString() const;
  uint64_t get_active_task_count() const;

 private:
  Queue<Task> tasks_;

  std::vector<std::thread> threads_;
  std::atomic_bool stop_ = false;
  std::exception_ptr ex_ptr_ = nullptr;
  std::condition_variable cond_;
  std::mutex mutex_;

  pid_t original_pid_;

  bool propagate_exception_ = false;

  std::atomic<uint64_t> active_task_count_{0};

  void main_loop() {
    while (!stop_) {
      executePendingTask(std::move(tasks_.pop()));
      if (--active_task_count_ == 0) {
        std::unique_lock lock(mutex_);
        cond_.notify_all();
      }
    }
  }
  void executePendingTask(Task&& task);
  void Init(const std::function<void()>& init_thread, uint64_t threads_number);
};

template <
    template <typename>
    typename Queue,
    typename Task,
    typename ThreadPolicy>
template <
    typename T,
    typename std::enable_if_t<std::is_same_v<T, MultiThreadPolicy>, bool>>
ThreadPoolBase<Queue, Task, ThreadPolicy>::ThreadPoolBase(
    bool propagate_exception,
    const std::function<void()>& init_thread,
    uint64_t threads_number)
    : propagate_exception_(propagate_exception) {
  Init(
      init_thread,
      threads_number == 0
          ? ThreadPolicy::GetNumThreads()
          : std::min(threads_number, ThreadPolicy::GetNumThreads()));
}

template <
    template <typename>
    typename Queue,
    typename Task,
    typename ThreadPolicy>
template <
    typename T,
    typename std::enable_if_t<std::is_same_v<T, move_only_function_void>, bool>>
void ThreadPoolBase<Queue, Task, ThreadPolicy>::enqueue(T&& task) {
  RethrowIfException();
  ++active_task_count_;
  tasks_.push(std::move(task));
}

template <
    template <typename>
    typename Queue,
    typename Task,
    typename ThreadPolicy>
template <
    class F,
    class... Args,
    typename T,
    typename std::enable_if_t<std::is_same_v<T, move_only_function_void>, bool>>
void ThreadPoolBase<Queue, Task, ThreadPolicy>::enqueue(F&& f, Args&&... args) {
  RethrowIfException();
  ++active_task_count_;
  auto task = [args = std::make_tuple(std::forward<Args>(args)...),
               func = std::move(f)]() mutable {
    std::apply([&](auto&&... x) { func(std::forward<Args>(x)...); }, args);
  };
  tasks_.push(std::move(task));
};

template <
    template <typename>
    typename Queue,
    typename Task,
    typename ThreadPolicy>
template <
    class F,
    class... Args,
    typename T,
    typename std::
        enable_if_t<std::is_same_v<T, std::packaged_task<void()>>, bool>>
std::future<void> ThreadPoolBase<Queue, Task, ThreadPolicy>::enqueue(
    F&& f,
    Args&&... args) {
  ++active_task_count_;
  auto packed_func = [args = std::make_tuple(std::forward<Args>(args)...),
                      func = std::move(f)]() mutable {
    std::apply([&](auto&&... x) { func(std::forward<Args>(x)...); }, args);
  };
  auto task = std::packaged_task<void()>(std::move(packed_func));
  auto res = task.get_future();
  tasks_.push(std::move(task));
  return res;
};

template <
    template <typename>
    typename Queue,
    typename Task,
    typename ThreadPolicy>
template <
    typename T,
    typename std::enable_if_t<std::is_same_v<T, move_only_function_void>, bool>>
void ThreadPoolBase<Queue, Task, ThreadPolicy>::waitWorkComplete() {
  RethrowIfException();
  if (active_task_count_ == 0 || stop_)
    return;

  // We try to detect case when process has been forked. In that case working
  // thread doesn't exist
  if (original_pid_ != getpid())
    return;

  {
    std::unique_lock lock(mutex_);
    cond_.wait(lock, [this] { return active_task_count_ == 0; });
  }

  RethrowIfException();
}

using SingleThreadPool =
    ThreadPoolBase<BlockingQueue, move_only_function_void, SingleThreadPolicy>;

using ThreadPool =
    ThreadPoolBase<BlockingQueue, move_only_function_void, MultiThreadPolicy>;

// This is deprecated version which has to be removed along with lazy execution
using SingleThreadPoolWithFutures = ThreadPoolBase<
    BlockingQueue,
    std::packaged_task<void()>,
    SingleThreadPolicy>;

} // namespace habana_helpers
