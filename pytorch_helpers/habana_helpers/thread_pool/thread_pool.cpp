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
    if (propagate_exception_) {
      ex_ptr_ = std::current_exception();
      PT_BRIDGE_WARN("Exception caught in thread: ", e.what());
    } else
      PT_BRIDGE_FATAL("Exception caught in thread: ", e.what());
  } catch (...) {
    if (propagate_exception_) {
      ex_ptr_ = std::current_exception();
      PT_BRIDGE_WARN("Exception caught in thread: unknown");
    } else
      PT_BRIDGE_FATAL("Exception caught in thread: unknown");
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

template <template <typename> typename Queue, typename Task>
uint64_t ThreadPoolBase<Queue, Task>::get_active_task_count() const {
  return active_task_count_.load();
}

template class ThreadPoolBase<BlockingQueue, move_only_function_void>;
template class ThreadPoolBase<BlockingQueue, std::packaged_task<void()>>;

} // namespace habana_helpers
