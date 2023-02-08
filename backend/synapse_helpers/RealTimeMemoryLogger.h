/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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