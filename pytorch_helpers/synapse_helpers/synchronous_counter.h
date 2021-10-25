/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 *******************************************************************************/
#pragma once

#include <atomic>
#include <condition_variable>
#include <ios>
#include <mutex>
#include <thread>

namespace synapse_helpers {
class synchronous_counter {
 public:
  void increment() {
    ++counter_;
  }
  void decrement() {
    --counter_;
    cv_.notify_one();
  }

  bool empty() {
    return counter_ == 0;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() { return counter_ == 0; });
  }

  template <class Period>
  bool wait_for(std::chrono::duration<int64_t, Period> wait_time) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cv_.wait_for(lock, wait_time, [this]() { return counter_ == 0; });
  }

 private:
  std::atomic<int> counter_{0};
  std::condition_variable cv_;
  std::mutex mtx_;
};

} // namespace synapse_helpers
