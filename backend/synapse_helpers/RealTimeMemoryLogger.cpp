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

#include "RealTimeMemoryLogger.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

#include "habana_helpers/logging.h"
#include "pool_allocator/CoalescedStringentPoolAllocator.h"

namespace synapse_helpers::realtime_logger {

PipeClient::PipeClient(const std::string& file_out) {
  if ((wfd_ = open(file_out.c_str(), O_WRONLY)) < 0)
    PT_BRIDGE_WARN("RealTimer Logger: open() error for read end");
}
PipeClient::~PipeClient() {
  close(wfd_);
}

void PipeClient::communicate(const std::vector<uint64_t>& msg) {
  if (wfd_ < 0)
    return;
  size_t size = msg.size();

  if (write(wfd_, &size, sizeof(size)) == -1)
    PT_BRIDGE_WARN("RealTimer Logger: write() error");
  if (write(wfd_, msg.data(), msg.size() * sizeof(uint64_t)) == -1)
    PT_BRIDGE_WARN("RealTimer Logger: write() error");
}

void RealTimeMeoryLogger::thread_loop() {
  std::vector<uint64_t> data;
  while (!stop_) {
    data.clear();
    allocator_->get_memory_mask(data);
    client_.communicate(data);
    std::this_thread::sleep_for(period_);
  }
}
} // namespace synapse_helpers::realtime_logger