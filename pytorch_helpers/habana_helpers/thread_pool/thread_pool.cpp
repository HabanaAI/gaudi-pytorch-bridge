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

ThreadPool::ThreadPool(size_t threads, QueueType qType) : m_stop(false) {
  m_tasks = Queue<std::function<void()>>::Create(
      qType, GET_ENV_FLAG_NEW(PT_HPU_THREAD_POOL_QUEUE_CAPACITY));
  for (size_t i = 0; i < threads; ++i)
    m_workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(this->m_queueMutex);
            this->m_condition.wait(lock, [this] {
              return this->m_stop || !this->m_tasks->empty();
            });
            if (this->m_stop && this->m_tasks->empty()) {
              this->has_queued_items.store(false);
              if (this->m_tasks) {
                delete this->m_tasks;
                this->m_tasks = NULL;
              }
              return;
            }
            task = std::move(this->m_tasks->front());
            this->m_tasks->pop();
        }

        // Run the task.
        try {
          task();
        } catch (const std::exception& e) {
          PT_BRIDGE_FATAL("Exception in launch thread pool task: ", e.what());
        } catch (...) {
          PT_BRIDGE_FATAL("Exception in launch thread pool task: unknown");
        }

          if (this->m_tasks->empty()) {
            this->has_queued_items.store(false);
          } else {
            this->has_queued_items.store(true);
          }
      }
    });
}

void ThreadPool::joinAllThreads() {
  for (std::thread& worker : m_workers) {
    worker.join();
  }
}

bool ThreadPool::inThreadPool() const {
  static thread_local std::thread::id tid = std::this_thread::get_id();
  for (auto& thread : m_workers) {
    if (thread.get_id() == tid) {
      return true;
    }
  }
  return false;
}

std::string ThreadPool::ToString() {
  std::stringstream ss;
  ss << "ThreadPool has_queued_items:" << has_queued_items;
  ss << " m_workers size:" << m_workers.size();
  ss << " m_tasks size:" << m_tasks->size();
  return ss.str();
}

// the destructor joins all threads
ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_stop = true;
  }
  m_condition.notify_all();
  joinAllThreads();
}
} // namespace habana_helpers
