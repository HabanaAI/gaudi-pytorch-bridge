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
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace synapse_helpers::pool_allocator {
class CoalescedStringentPooling;
}

namespace synapse_helpers::realtime_logger {

class PipeClient {
 public:
  explicit PipeClient(const std::string& out = "/tmp/pipe_memory_data.txt");
  ~PipeClient();

  void communicate(const std::vector<uint64_t>& msg);

 private:
  int wfd_;
};

class RealTimeMeoryLogger {
 public:
  RealTimeMeoryLogger(
      const synapse_helpers::pool_allocator::CoalescedStringentPooling*
          allocator)
      : thread_(&RealTimeMeoryLogger::thread_loop, this),
        stop_(false),
        allocator_(allocator) {}
  ~RealTimeMeoryLogger() {
    stop_ = true;
    thread_.join();
  }
  void thread_loop();

 private:
  static constexpr auto period_ = std::chrono::milliseconds(300);
  PipeClient client_;
  std::thread thread_;
  std::atomic<bool> stop_;
  const synapse_helpers::pool_allocator::CoalescedStringentPooling* allocator_;
};

} // namespace synapse_helpers::realtime_logger