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
#include <sys/sysinfo.h>
#include <iostream>

#include "backend/synapse_helpers/env_flags.h"
#include "thread_pool.h"

namespace habana_helpers {

template <template <typename> typename Queue>
ThreadPoolBase<Queue>::ThreadPoolBase(bool propagate_exception)
    : stop_(false),
      ex_ptr_(nullptr),
      propagate_exception_(propagate_exception) {
  thread_ = std::thread(&ThreadPoolBase<Queue>::main_loop, this);
}

template <template <typename> typename Queue>
ThreadPoolBase<Queue>::~ThreadPoolBase() {
  // set flag to true to break main loop in the acc thread
  tasks_.push(std::packaged_task<void()>{[this]() { stop_ = true; }});
  try {
    thread_.join();
  } catch (const std::exception& ex) {
    PT_BRIDGE_WARN("Exception in pool destructor: ", ex.what());
  }
}

template <template <typename> typename Queue>
void ThreadPoolBase<Queue>::waitWorkComplete() {
  auto task = std::packaged_task<void()>([]() {});
  auto work_compelete = task.get_future();
  tasks_.push(std::move(task));
  work_compelete.wait();
}

template <template <typename> typename Queue>
void ThreadPoolBase<Queue>::executePendingTask(Task&& task) {
  try {
    task();
  } catch (const std::exception& e) {
    if (propagate_exception_)
      ex_ptr_ = std::current_exception();
    else
      PT_BRIDGE_FATAL("Exception in launch thread pool task: ", e.what());
  } catch (...) {
    if (propagate_exception_)
      ex_ptr_ = std::current_exception();
    else
      PT_BRIDGE_FATAL("Exception in launch thread pool task: unknown");
  }
}

template <template <typename> typename Queue>
void ThreadPoolBase<Queue>::rethrowIfException() {
  if (ex_ptr_) {
    auto ex_ptr = ex_ptr_;
    ex_ptr_ = nullptr;
    std::rethrow_exception(ex_ptr);
  }
}

template <template <typename> typename Queue>
std::string ThreadPoolBase<Queue>::ToString() const {
  std::stringstream ss;
  ss << "ThreadPool m_tasks size:" << tasks_.size();
  return ss.str();
}

template class ThreadPoolBase<BlockingQueue>;

} // namespace habana_helpers
