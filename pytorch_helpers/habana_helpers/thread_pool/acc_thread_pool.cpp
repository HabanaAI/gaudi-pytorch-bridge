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
#include <ATen/Parallel.h>
#include <c10/util/thread_name.h>
#include <future>

#include "habana_helpers/logging.h"
#include "habana_lazy/lazy_graph_hash_disabler.h"
#include "pytorch_helpers/habana_helpers/thread_pool/acc_thread_pool.h"

namespace habana_lazy {

template <typename T>
/**
 * clang16 complains AccThreadPoolFast cannot be instantiated with LockFreeQueue
 * as it has multiple template parameters (even if latter has default). This is
 * to workaround this.
 */
struct LockFreeQueueDefaultSize : public LockFreeQueue<T> {
  using LockFreeQueue<T>::LockFreeQueue;
};

std::unique_ptr<AccThreadPoolBase> CreateAccThreadPool() {
  if (GET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING))
    return std::make_unique<AccNoThread>();
  else {
    int thread_ver = GET_ENV_FLAG_NEW(PT_HPU_ACC_THREAD_VERSION);
    switch (thread_ver) {
      case 0:
        return std::make_unique<AccThreadPool>();
      case 1:
        return std::make_unique<AccThreadPoolFast<BlockingQueue>>();
      default:
        return std::make_unique<AccThreadPoolFast<LockFreeQueueDefaultSize>>();
    }
  }
}

// ////////////////////////////////////////////////

void AccNoThread::run(AccTask&& func) {
  std::unique_lock<std::mutex> lock(mutex_);
  tasks_.emplace(std::move(func));
}

void AccNoThread::waitWorkComplete() {
  // If task_in_progress_ is true then we reentered AccThread - this is
  // situation we want to avoid
  HABANA_ASSERT(task_in_progress_ == false);
  task_in_progress_ = true;

  while (!tasks_.empty()) {
    std::unique_lock<std::mutex> lock(mutex_);
    AccTask task = std::move(tasks_.front());
    tasks_.pop();
    lock.unlock();
    DisableRunningHashUpdates disable;
    task();
  }

  task_in_progress_ = false;
}

void AccNoThread::discardPendingTasks() {
  std::unique_lock<std::mutex> lock(mutex_);
  std::queue<AccTask> empty_queue;
  tasks_.swap(empty_queue);
}

bool AccNoThread::inAccThreadContext() const {
  return task_in_progress_;
}

// //////////////////////////////////////////////

AccThreadPool::AccThreadPool()
    : stop_(false), task_count_(0), ex_ptr_(nullptr) {
  auto init_thread = []() {
    c10::setThreadName("AccThreadPool");
    at::init_num_threads();
  };

  thread_ = std::thread([this, init_thread]() {
    init_thread();
    this->main_loop();
  });
}

AccThreadPool::~AccThreadPool() {
  // set flag to true to break main loop in the acc thread
  stop_ = true;

  try {
    thread_.join();
  } catch (const std::exception& ex) {
    PT_BRIDGE_WARN("Exception in acc thread pool desctructor: ", ex.what());
  }
}

bool AccThreadPool::inThreadPool() const {
  return thread_.get_id() == std::this_thread::get_id();
}

bool AccThreadPool::inAccThreadContext() const {
  return inThreadPool();
}

void AccThreadPool::run(std::function<void()>&& func) {
  checkNoException();

  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(std::move(func));
  }

  ++task_count_;
}

void AccThreadPool::waitWorkComplete() {
  while (task_count_ > 0)
    std::this_thread::yield();

  checkNoException();
}

void AccThreadPool::executePendingTask() {
  std::unique_lock<std::mutex> lock(mutex_);

  if (tasks_.empty()) {
    return;
  }

  {
    AccTask task = std::move(tasks_.front());
    tasks_.pop();
    lock.unlock();

    // Run the task.
    try {
      DisableRunningHashUpdates disable;
      task();
    } catch (...) {
      ex_ptr_ = std::current_exception();
    }
  }

  --task_count_;
}

void AccThreadPool::discardPendingTasks() {
  HABANA_ASSERT(false, "discardPendingTasks supported only for sync mode");
}

void AccThreadPool::main_loop() {
  while (!stop_) {
    // wait until there are available tasks in the queue or
    // accumulation thread pool is destructured
    while (task_count_ == 0 && !stop_) {
    }

    // break if accumulation thread pool is destructured
    if (stop_) {
      break;
    }

    executePendingTask();
  } // while !stop_
  task_count_ = 0;
}

void AccThreadPool::checkNoException() {
  std::unique_lock<std::mutex> lock(mutex_);
  try {
    if (ex_ptr_) {
      std::rethrow_exception(ex_ptr_);
    }
  } catch (const std::exception& ex) {
    ex_ptr_ = nullptr;
    PT_BRIDGE_FATAL(
        "Exception in acc thread pool task has been thrown: ", ex.what());
  }
}

// /////////////////////////////////////////////////////////////////////
template <template <typename> typename Queue>
AccThreadPoolFast<Queue>::AccThreadPoolFast() : stop_(false), ex_ptr_(nullptr) {
  auto init_thread = []() {
    c10::setThreadName("AccThreadPool");
    at::init_num_threads();
  };
  thread_ = std::thread([this, init_thread]() {
    init_thread();
    this->main_loop();
  });
}

template <template <typename> typename Queue>
AccThreadPoolFast<Queue>::~AccThreadPoolFast() {
  // set flag to true to break main loop in the acc thread
  tasks_.push({[this]() { stop_ = true; }});
  try {
    thread_.join();
  } catch (const std::exception& ex) {
    PT_BRIDGE_WARN("Exception in acc thread pool destructor: ", ex.what());
  }
}

template <template <typename> typename Queue>
bool AccThreadPoolFast<Queue>::inThreadPool() const {
  return thread_.get_id() == std::this_thread::get_id();
}

template <template <typename> typename Queue>
bool AccThreadPoolFast<Queue>::inAccThreadContext() const {
  return inThreadPool();
}

template <template <typename> typename Queue>
void AccThreadPoolFast<Queue>::run(std::function<void()>&& func) {
  checkNoException();
  tasks_.push(std::move(func));
}

template <template <typename> typename Queue>
void AccThreadPoolFast<Queue>::waitWorkComplete() {
  checkNoException();

  std::promise<void> last_task;
  std::future<void> work_compelete = last_task.get_future();
  tasks_.push({[&last_task]() { last_task.set_value(); }, true});
  work_compelete.wait();

  checkNoException();
}

template <template <typename> typename Queue>
void AccThreadPoolFast<Queue>::executePendingTask(
    std::function<void()>&& task) {
  try {
    DisableRunningHashUpdates disable;
    task();
  } catch (...) {
    ex_ptr_ = std::current_exception();
    stop_ = true;
    return;
  }
}

template <template <typename> typename Queue>
void AccThreadPoolFast<Queue>::discardPendingTasks() {
  HABANA_ASSERT(false, "discardPendingTasks supported only for sync mode");
}

template <template <typename> typename Queue>
void AccThreadPoolFast<Queue>::main_loop() {
  while (!stop_) {
    executePendingTask(std::move(tasks_.pop().fun_));
  }
  // in case of exception has been thrown
  while (!tasks_.empty()) {
    auto task = std::move(tasks_.pop());
    if (task.intra_task_)
      task.fun_();
  }
}

template <template <typename> typename Queue>
void AccThreadPoolFast<Queue>::checkNoException() {
  try {
    if (ex_ptr_) {
      std::rethrow_exception(ex_ptr_);
    }
  } catch (const std::exception& ex) {
    PT_BRIDGE_FATAL(
        "Exception in acc thread pool task has been thrown: ", ex.what());
  } catch (...) {
    PT_BRIDGE_FATAL("Exception in acc thread pool task has been thrown");
  }
  ex_ptr_ = nullptr;
}

template class AccThreadPoolFast<LockFreeQueueDefaultSize>;
template class AccThreadPoolFast<BlockingQueue>;

} // namespace habana_lazy
