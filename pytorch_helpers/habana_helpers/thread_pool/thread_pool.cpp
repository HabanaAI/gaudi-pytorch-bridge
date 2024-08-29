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

template <template <typename> typename Queue, typename Task>
ThreadPoolBase<Queue, Task>::ThreadPoolBase(bool propagate_exception)
    : stop_(false),
      ex_ptr_(nullptr),
      propagate_exception_(propagate_exception) {
  thread_ = std::thread(&ThreadPoolBase<Queue, Task>::main_loop, this);
  original_pid_ = getpid();
}

template <template <typename> typename Queue, typename Task>
ThreadPoolBase<Queue, Task>::~ThreadPoolBase() {
  // set flag to true to break main loop in the thread
  ++active_task_count_;
  tasks_.push(Task{[this]() { stop_ = true; }});
  try {
    thread_.join();
  } catch (const std::exception& ex) {
    PT_BRIDGE_WARN("Exception in pool destructor: ", ex.what());
  }
}

template <template <typename> typename Queue, typename Task>
void ThreadPoolBase<Queue, Task>::executePendingTask(Task&& task) {
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

template <template <typename> typename Queue, typename Task>
void ThreadPoolBase<Queue, Task>::RethrowIfException() {
  if (ex_ptr_) {
    auto ex_ptr = ex_ptr_;
    ex_ptr_ = nullptr;
    std::rethrow_exception(ex_ptr);
  }
}

template <template <typename> typename Queue, typename Task>
std::string ThreadPoolBase<Queue, Task>::ToString() const {
  return std::string("ThreadPool m_tasks size: ") +
      std::to_string(tasks_.size());
}

template class ThreadPoolBase<BlockingQueue, move_only_function_void>;
template class ThreadPoolBase<BlockingQueue, std::packaged_task<void()>>;

} // namespace habana_helpers
