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
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "pytorch_helpers/habana_helpers/thread_queue.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

namespace habana_helpers {

/**
 * @brief Used to run different compares in different threads
 *
 */
class ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  void joinAllThreads();
  bool inThreadPool() const;
  std::thread::id get_id(size_t worker);
  ~ThreadPool();
  bool m_stop;
  std::atomic<bool> has_work{false};
  std::atomic<bool> has_queued_items{false};

 private:
  std::vector<std::thread> m_workers;
  Queue<std::function<void()>>* m_tasks;
  // synchronization
  std::mutex m_queueMutex;
  std::condition_variable m_condition;
};

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    if (!(GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
          (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2))) {
      std::lock_guard<std::mutex> lock(m_queueMutex);
      // don't allow enqueueing after stopping the pool
      if (m_stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      m_tasks->emplace([task]() { (*task)(); });
    } else {
      if (m_stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      m_tasks->emplace([task]() { (*task)(); });
      has_work.store(true);
    }
  }
  if (!(GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT) &&
        (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2))) {
    m_condition.notify_one();
  }
  return res;
}

inline std::thread::id ThreadPool::get_id(size_t worker) {
  std::thread::id tid(-1);
  if (worker < m_workers.size()) {
    tid = m_workers[worker].get_id();
  }
  return tid;
}

} // namespace habana_helpers
