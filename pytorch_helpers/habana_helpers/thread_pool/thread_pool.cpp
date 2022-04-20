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
          while (!this->has_work.load()) {
            if (this->m_stop && !this->has_work.load()) {
              return;
            }
          }
          if (this->m_stop && !this->has_work.load())
            return;
          task = std::move(this->m_tasks.front());
          this->m_tasks.pop();
        }

        task();
        this->has_work.store(false);
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
  joinAllThreads();
}
} // namespace habana_helpers
