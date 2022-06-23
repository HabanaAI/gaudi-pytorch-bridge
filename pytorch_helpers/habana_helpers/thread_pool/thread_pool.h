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
  std::thread::id get_id(size_t worker);
  ~ThreadPool();

 private:
  std::vector<std::thread> m_workers;
  std::queue<std::function<void()>> m_tasks;
  // synchronization
  std::mutex m_queueMutex;
  std::condition_variable m_condition;
  bool m_stop;
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
    std::lock_guard<std::mutex> lock(m_queueMutex);

    // don't allow enqueueing after stopping the pool
    if (m_stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    m_tasks.emplace([task]() { (*task)(); });
  }
  m_condition.notify_one();
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
