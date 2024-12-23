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
#include "pytorch_helpers/habana_helpers/python_utils.h"
#include "thread_pool.h"

namespace habana_helpers {

namespace {
#if defined(__linux__)
#include <sched.h>

uint64_t GetAvailableThreads() {
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);

  // Get the affinity mask for the current process
  auto result = sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpuSet);
  HABANA_ASSERT(result == 0)

  int threads_count = 0;
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &cpuSet)) {
      ++threads_count;
    }
  }

  return threads_count;
}
#else
uint64_t GetAvailableThreads() {
  auto num_threads = std::thread::hardware_concurrency();
  return num_threads == 0 ? 1 : num_threads;
}
#endif
} // namespace

ThreadPool::ThreadPool() : stop_(false) {
  for (uint64_t i = 0; i < GetAvailableThreads(); ++i)
    threads_.emplace_back(&ThreadPool::main_loop, this);
}

ThreadPool::~ThreadPool() {
  stop_ = true;
  for (size_t i = 0; i < threads_.size(); ++i)
    tasks_.push(Task{[this]() {}});

  for (auto& thread : threads_)
    thread.join();
}

void ThreadPool::executePendingTask(Task&& task) {
  ++active_count_;
  try {
    task();
  } catch (const std::exception& e) {
    PT_BRIDGE_FATAL("Exception caught in thread: ", e.what());
  } catch (...) {
    PT_BRIDGE_FATAL("Exception caught in thread: unknown");
  }
  --active_count_;
}

template <template <typename> typename Queue, typename Task>
SingleThreadPoolBase<Queue, Task>::SingleThreadPoolBase(
    bool propagate_exception,
    uint64_t queue_capacity,
    const std::function<void()>& init_thread)
    : stop_(false),
      ex_ptr_(nullptr),
      propagate_exception_(propagate_exception),
      queue_capacity_(queue_capacity) {
  thread_ = std::thread([this, init_thread]() {
    if (init_thread)
      init_thread();
    this->main_loop();
  });
  original_pid_ = getpid();
}

template <template <typename> typename Queue, typename Task>
SingleThreadPoolBase<Queue, Task>::~SingleThreadPoolBase() {
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
void SingleThreadPoolBase<Queue, Task>::executePendingTask(Task&& task) {
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
void SingleThreadPoolBase<Queue, Task>::throttleIfNeeded() {
  if (queue_capacity_ > 0 && active_task_count_ >= queue_capacity_) {
    // throttle only when queue capacity is limited
    // and active tasks exceeds the configured capacity
    auto throttle_limit = queue_capacity_ / 2;
    // Release GIL if going to wait (remove once SW-160978 is fixed)
    habana_helpers::AutoNoGIL gil_release;
    while (active_task_count_ > throttle_limit) {
      std::this_thread::yield();
    }
  }
}

template <template <typename> typename Queue, typename Task>
void SingleThreadPoolBase<Queue, Task>::RethrowIfException() {
  if (ex_ptr_) {
    auto ex_ptr = ex_ptr_;
    ex_ptr_ = nullptr;
    std::rethrow_exception(ex_ptr);
  }
}

template <template <typename> typename Queue, typename Task>
std::string SingleThreadPoolBase<Queue, Task>::ToString() const {
  return std::string("ThreadPool m_tasks size: ") +
      std::to_string(tasks_.size());
}

template <template <typename> typename Queue, typename Task>
uint64_t SingleThreadPoolBase<Queue, Task>::get_active_task_count() const {
  return active_task_count_.load();
}

template class SingleThreadPoolBase<BlockingQueue, move_only_function_void>;
template class SingleThreadPoolBase<BlockingQueue, std::packaged_task<void()>>;

} // namespace habana_helpers
