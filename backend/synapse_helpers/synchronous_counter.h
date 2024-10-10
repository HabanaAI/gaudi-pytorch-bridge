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
