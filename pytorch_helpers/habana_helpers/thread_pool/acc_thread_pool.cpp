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

#include "habana_lazy/lazy_graph_hash_disabler.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/thread_pool/acc_thread_pool.h"

namespace habana_lazy {

thread_local bool AccThreadPool::task_in_progress_{false};

AccThreadPool::AccThreadPool()
    : threads_(1), running_(true), task_count_(0), ex_ptr_(nullptr) {
  auto init_thread = []() {
    c10::setThreadName("AccThreadPool");
    at::init_num_threads();
  };

  if (!GET_ENV_FLAG_NEW(PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING)) {
    for (std::size_t i = 0; i < threads_.size(); ++i) {
      threads_[i] = std::thread([this, init_thread]() {
        init_thread();
        this->main_loop();
      });
    }
  } else {
    running_ = false;
  }
}

AccThreadPool::~AccThreadPool() {
  // set flag to false to break main loop in the acc thread
  running_ = false;

  for (auto& t : threads_) {
    try {
      t.join();
    } catch (const std::exception& ex) {
      PT_BRIDGE_WARN("Exception in acc thread pool desctructor: ", ex.what());
    }
  }
}

bool AccThreadPool::inThreadPool() const {
  static thread_local std::thread::id tid = std::this_thread::get_id();
  for (auto& thread : threads_) {
    if (thread.get_id() == tid) {
      return true;
    }
  }
  return false;
}

bool AccThreadPool::inAccThreadContext() const {
  return inThreadPool() || task_in_progress_;
}

void AccThreadPool::run(std::function<void()>&& func) {
  if (threads_.size() == 0) {
    throw std::runtime_error("No threads to run a task");
  }
  checkNoException();

  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(std::move(func));
  }

  ++task_count_;
}

void AccThreadPool::waitWorkComplete() {
  while (task_count_ > 0) {
    if (!running_) {
      executePendingTask();
    }
  }

  checkNoException();
}

void AccThreadPool::executePendingTask() {
  std::unique_lock<std::mutex> lock(mutex_);

  if (tasks_.empty()) {
    return;
  }

  AccTask task = std::move(tasks_.front());
  tasks_.pop();
  lock.unlock();

  // If task_in_progress_ is true then we reentered AccThread - this is
  // situation we want to avoid
  HABANA_ASSERT(task_in_progress_ == false);
  task_in_progress_ = true;

  // Run the task.
  try {
    DisableRunningHashUpdates disable;
    task();
  } catch (...) {
    ex_ptr_ = std::current_exception();
    running_ = false;
    this->discardPendingTasks();
    return;
  }

  task_in_progress_ = false;
  --task_count_;
}

void AccThreadPool::discardPendingTasks() {
  HABANA_ASSERT(running_ == false);
  std::queue<AccTask> empty_queue;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.swap(empty_queue);
  }
  task_count_ = 0;
}

void AccThreadPool::main_loop() {
  while (running_) {
    // wait until there are available tasks in the queue or
    // accumulation thread pool is destructured
    while (task_count_ == 0 && running_) {
    }

    // break if accumulation thread pool is destructured
    if (!running_) {
      break;
    }

    executePendingTask();
  } // while running_
  checkNoException();
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

} // namespace habana_lazy