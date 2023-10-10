#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

/**
 * @brief Used to run different compares in different threads
 *
 */
class ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type>;
  ~ThreadPool();

 private:
  std::vector<std::thread> m_workers;
  std::queue<std::function<void()>> m_tasks;

  // synchronization
  std::mutex m_queueMutex;
  std::condition_variable m_condition;
  bool m_stop;
};

inline ThreadPool::ThreadPool(size_t threads) : m_stop(false) {
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

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
  using return_type = typename std::invoke_result<F, Args...>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(m_queueMutex);

    // don't allow enqueueing after stopping the pool
    if (m_stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    m_tasks.emplace([task]() { (*task)(); });
  }
  m_condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(m_queueMutex);
    m_stop = true;
  }
  m_condition.notify_all();
  for (std::thread& worker : m_workers)
    worker.join();
}