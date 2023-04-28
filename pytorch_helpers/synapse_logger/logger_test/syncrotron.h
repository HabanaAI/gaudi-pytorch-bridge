/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <sys/syscall.h>
#include <sys/types.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
inline uint64_t NowMicros() {
  static std::chrono::time_point<std::chrono::high_resolution_clock> t0 =
      std::chrono::high_resolution_clock::now();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - t0)
          .count());
}

inline int get_current_tid() {
  return syscall(__NR_gettid);
}

class syncrotron {
 public:
  std::atomic<unsigned> step;
  std::condition_variable sync_cv;
  std::mutex mutex;
  std::vector<std::thread> threads;
  using sync_lock = std::unique_lock<std::mutex>;
  syncrotron() {
    threads.reserve(16);
  }

  template <typename... Args>
  void add_proc(Args&&... args) {
    threads.emplace_back(std::forward<Args>(args)...);
  }

  void sync(int set_step) {
    sync_lock lock(mutex);
    std::clog << NowMicros() << ": tid " << get_current_tid() << " waiting for "
              << set_step - 1 << " step " << step << std::endl;
    if (step < set_step - 1)
      sync_cv.wait(lock, [this, set_step]() { return step == set_step - 1; });
    step = set_step;
    std::clog << NowMicros() << " step " << step << std::endl;
    lock.unlock();
    sync_cv.notify_all();
  }
  void start() {
    sync_cv.notify_all();
  }
  ~syncrotron() {
    for (auto& t : threads) {
      t.join();
    }
  }
};
