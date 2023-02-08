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