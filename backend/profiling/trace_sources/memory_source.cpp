
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

#include "memory_source.h"
#include <syscall.h>
#include <unistd.h>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>
#include "backend/profiling/trace_sources/sources.h"

namespace habana {
namespace profile {

struct MemoryLogger : public TraceSource {
  MemoryLogger() = default;
  bool enabled() {
    return enabled_;
  }
  static MemoryLogger& instance() {
    static MemoryLogger source;
    return source;
  }
  void start(TraceSink&) {
    enabled_ = true;
  }
  void stop() {
    enabled_ = false;
  }
  void extract(TraceSink& output) {
    pid_t tid = syscall(__NR_gettid);
    pid_t pid = getpid() + offset_;
    std::lock_guard<std::mutex> lg{m};
    for (const auto& event : events_) {
      output.addMemoryEvent(
          pid,
          tid,
          event.time,
          event.addr,
          event.bytes,
          0,
          1,
          event.total_allocated,
          event.total_reserved);
    }
    output.addDevice("Memory Logs", pid);
    events_.clear();
  }
  TraceSourceVariant get_variant() {
    return TraceSourceVariant::MEMORY_LOGS;
  }
  void set_offset(unsigned offset) {
    offset_ = offset;
  }

  void recordAllocation(
      uint64_t addr,
      int64_t size,
      uint64_t total_allocated,
      uint64_t total_reserved) {
    if (enabled_ && size > 0) {
      int64_t dtime = nowNanos();
      std::lock_guard<std::mutex> lg{m};
      ptrs_.emplace(addr, size);
      events_.emplace_back(dtime, addr, size, total_allocated, total_reserved);
    }
  }
  void recordDeallocation(
      uint64_t addr,
      uint64_t total_allocated,
      uint64_t total_reserved) {
    if (enabled_) {
      std::lock_guard<std::mutex> lg{m};
      auto it = ptrs_.find(addr);
      if (it != ptrs_.end()) {
        int64_t dtime = nowNanos();
        int64_t bytes{static_cast<int64_t>(it->second) * -1};
        events_.emplace_back(
            dtime, addr, bytes, total_allocated, total_reserved);
        ptrs_.erase(it);
      }
    }
  }
  uint64_t nowNanos() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }

 private:
  struct Event {
    int64_t time;
    uint64_t addr;
    int64_t bytes;
    uint64_t total_allocated;
    uint64_t total_reserved;

    Event(
        int64_t time,
        uint64_t addr,
        int64_t bytes,
        uint64_t total_allocated,
        uint64_t total_reserved)
        : time(time),
          addr(addr),
          bytes(bytes),
          total_allocated(total_allocated),
          total_reserved(total_reserved) {}
  };
  std::unordered_map<uint64_t, uint64_t> ptrs_;
  std::mutex m{};
  std::deque<Event> events_;
  bool enabled_{false};
  unsigned offset_{};
};

MemorySource::~MemorySource() {}
void MemorySource::start(TraceSink& sink) {
  MemoryLogger::instance().start(sink);
}
void MemorySource::stop() {
  MemoryLogger::instance().stop();
}
void MemorySource::extract(TraceSink& output) {
  MemoryLogger::instance().extract(output);
}
TraceSourceVariant MemorySource::get_variant() {
  return MemoryLogger::instance().get_variant();
}
void MemorySource::set_offset(unsigned offset) {
  MemoryLogger::instance().set_offset(offset);
}

namespace memory {
void recordAllocation(
    uint64_t addr,
    uint64_t size,
    uint64_t total_allocated,
    uint64_t total_reserved) {
  MemoryLogger::instance().recordAllocation(
      addr, size, total_allocated, total_reserved);
};
void recordDeallocation(
    uint64_t addr,
    uint64_t total_allocated,
    uint64_t total_reserved) {
  MemoryLogger::instance().recordDeallocation(
      addr, total_allocated, total_reserved);
}
bool enabled() {
  return MemoryLogger::instance().enabled();
}
}; // namespace memory

}; // namespace profile
}; // namespace habana