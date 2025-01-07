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
