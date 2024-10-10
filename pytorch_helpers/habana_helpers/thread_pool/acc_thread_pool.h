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
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana_lazy {

class AccThreadPoolBase {
 public:
  using AccTask = std::function<void()>;
  virtual void run(AccTask&& func) = 0;
  virtual void waitWorkComplete() = 0;
  virtual void discardPendingTasks() = 0;
  virtual bool inAccThreadContext() const = 0;
  virtual ~AccThreadPoolBase() = default;
};

// Synchronous mode of AccThreadPool
class AccNoThread final : public AccThreadPoolBase {
 public:
  void run(AccTask&& func) override;
  void waitWorkComplete() override;
  void discardPendingTasks() override;
  bool inAccThreadContext() const override;

 private:
  std::queue<AccTask> tasks_;
  std::atomic_bool task_in_progress_{false};
  std::mutex mutex_;
};

std::unique_ptr<AccThreadPoolBase> CreateAccThreadPool();

class AccThreadPool final : public AccThreadPoolBase {
 public:
  AccThreadPool();
  ~AccThreadPool();

  void run(std::function<void()>&& func) override;
  void waitWorkComplete() override;
  void discardPendingTasks() override;
  bool inAccThreadContext() const override;

 private:
  bool inThreadPool() const;
  std::queue<AccTask> tasks_;
  std::thread thread_;
  std::mutex mutex_;
  std::atomic_bool stop_;
  std::atomic<std::size_t> task_count_;
  std::exception_ptr ex_ptr_;

  // Check if no exception has been thrown by any task_. If excpetion occured
  // then rethrow it in the main thread.
  void checkNoException();
  void main_loop();
  void executePendingTask();
};

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
  void pop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    item = std::move(queue_.front());
    queue_.pop();
  }
  void top(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    item = queue_.front();
  }
  void push(const T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(item);
    lock.unlock();
    cond_.notify_one();
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

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};
template <typename T, size_t max_size = 8192>
class LockFreeQueue {
 public:
  T pop() {
    // only one consumer
    while (empty()) {
      std::this_thread::yield();
    };
    tail_ = (tail_ + 1) % max_size;
    return std::move(queue_[tail_]);
  }
  void push(T&& item) {
    // only one producer
    HABANA_ASSERT(enter_ == false);
    enter_ = true;
    size_t new_h = (head_ + 1) % max_size;
    HABANA_ASSERT(
        tail_ != new_h, "Limit of acc thread tasks has been exceeded");
    queue_[new_h] = std::forward<T>(item);
    head_ = new_h;
    enter_ = false;
  }
  bool empty() {
    return head_ == tail_;
  }

 private:
  std::atomic<size_t> head_ = 0;
  std::atomic<size_t> tail_ = 0;
  std::atomic<bool> enter_ = false;
  std::array<T, max_size> queue_;
};

// AccThreadPoolFast<LockFreeQueue> - the fastest solution based on spinning and
// doesn't use mutexes. However, could consume more CPU time
// AccThreadPoolFast<BlockingQueue> - slower solution based on mutex but use
// less CPU times that could be usefull for CPU bound setup
template <template <typename> typename Queue>
class AccThreadPoolFast final : public AccThreadPoolBase {
 public:
  AccThreadPoolFast();
  ~AccThreadPoolFast();

  void run(AccTask&& func) override;
  void waitWorkComplete() override;
  void discardPendingTasks() override;
  bool inAccThreadContext() const override;

 private:
  struct AccTaskInternal {
    bool intra_task_ = false;
    std::function<void()> fun_ = []() {};
    AccTaskInternal(std::function<void()>&& fun, bool intra_task = false)
        : intra_task_(intra_task), fun_(std::move(fun)){};
    AccTaskInternal(){};
  };

  bool inThreadPool() const;
  Queue<AccTaskInternal> tasks_;

  std::thread thread_;
  std::atomic_bool stop_;
  std::exception_ptr ex_ptr_;

  // Check if no exception has been thrown by any task_. If excpetion occured
  // then rethrow it in the main thread.
  void checkNoException();
  void main_loop();
  void executePendingTask(std::function<void()>&& task);
};
} // namespace habana_lazy