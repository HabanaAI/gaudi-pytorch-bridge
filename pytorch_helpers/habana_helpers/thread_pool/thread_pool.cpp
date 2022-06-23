/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include "thread_pool.h"

namespace habana_helpers {

ThreadPool::ThreadPool(size_t threads) : m_stop(false) {
  for (size_t i = 0; i < threads; ++i)
    m_workers.emplace_back([this] {
      for (;;) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->m_queueMutex);
          this->m_condition.wait(
              lock, [this] { return this->m_stop || !this->m_tasks.empty(); });
          if (this->m_stop && this->m_tasks.empty())
            return;
          task = std::move(this->m_tasks.front());
          this->m_tasks.pop();
        }

        task();
      }
    });
}

void ThreadPool::joinAllThreads() {
  for (std::thread& worker : m_workers) {
    worker.join();
  }
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
