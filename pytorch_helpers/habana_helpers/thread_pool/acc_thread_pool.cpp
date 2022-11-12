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
#include "pytorch_helpers/habana_helpers/thread_pool/acc_thread_pool.h"
#include <ATen/Parallel.h>
#include <c10/util/thread_name.h>
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana_lazy {

AccThreadPool::AccThreadPool() : threads_(1), running_(true), task_count_(0) {
  auto init_thread = []() {
    c10::setThreadName("AccThreadPool");
    at::init_num_threads();
  };

  for (std::size_t i = 0; i < threads_.size(); ++i) {
    threads_[i] = std::thread([this, init_thread]() {
      init_thread();
      this->main_loop();
    });
  }
}

AccThreadPool::~AccThreadPool() {
  // set flag to false to break main loop in the acc thread
  running_ = false;

  for (auto& t : threads_) {
    try {
      t.join();
    } catch (const std::exception&) {
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

void AccThreadPool::run(std::function<void()>&& func) {
  if (threads_.size() == 0) {
    throw std::runtime_error("No threads to run a task");
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);
    tasks_.emplace(std::move(func));
  }

  ++task_count_;
}

void AccThreadPool::waitWorkComplete() {
  while (task_count_ > 0) {
  }
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

    std::unique_lock<std::mutex> lock(mutex_);
    if (tasks_.empty()) {
      continue;
    }

    {
      AccTask task = std::move(tasks_.front());
      tasks_.pop();
      lock.unlock();

      // Run the task.
      try {
        task();
      } catch (const std::exception& e) {
        PT_BRIDGE_FATAL("Exception in acc thread pool task: ", e.what());
      } catch (...) {
        PT_BRIDGE_FATAL("Exception in acc thread pool task: unknown");
      }
    }

    --task_count_;
  } // while running_
}

} // namespace habana_lazy