/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */

#include "bridge_logs_source.h"
#include <syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include "pytorch_helpers/habana_helpers/logging.h"

namespace {
uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}
} // namespace

namespace habana {
namespace profile {

struct BridgeLogger : public TraceSource {
  BridgeLogger() = default;
  ~BridgeLogger() override = default;
  void log(std::string_view id, bool is_begin) {
    if (enabled()) {
      int64_t dtime = NowMicros();
      pid_t tid = syscall(__NR_gettid);
      std::string event_id{id};
      std::lock_guard<std::mutex> lg{m};
      events_.emplace_back(std::move(event_id), dtime, tid, is_begin);
    }
  }
  bool enabled() {
    return enabled_;
  }
  static BridgeLogger& instance() {
    static BridgeLogger source;
    return source;
  }
  void start() {
    enabled_ = true;
  }
  void stop() {
    enabled_ = false;
  }
  void extract(TraceSink& output) {
    pid_t pid = getpid() + offset_;
    std::lock_guard<std::mutex> lg{m};
    for (const auto& event : events_) {
      output.addActivity(
          {event.name, {}, ActivityType::RUNTIME, pid, event.tid},
          {},
          event.time,
          event.begin);
    }
    output.addDevice("Bridge Logs", pid);
    events_.clear();
  }
  TraceSourceVariant get_variant() {
    return TraceSourceVariant::BRIDGE_LOGS;
  }
  void set_offset(unsigned offset) {
    offset_ = offset;
  }

 private:
  struct Event {
    std::string name;
    int64_t time;
    pid_t tid;
    bool begin;
    Event(std::string&& name, int64_t time, pid_t tid, bool begin)
        : name(std::move(name)), time(time), tid(tid), begin(begin) {}
  };
  std::deque<Event> events_;
  std::atomic<bool> enabled_{false};
  unsigned offset_{};
  std::mutex m{};
};

BridgeLogsSource::~BridgeLogsSource() {}

void BridgeLogsSource::start() {
  BridgeLogger::instance().start();
}
void BridgeLogsSource::stop() {
  BridgeLogger::instance().stop();
}
void BridgeLogsSource::extract(TraceSink& output) {
  BridgeLogger::instance().extract(output);
}

TraceSourceVariant BridgeLogsSource::get_variant() {
  return BridgeLogger::instance().get_variant();
}
void BridgeLogsSource::set_offset(unsigned offset) {
  BridgeLogger::instance().set_offset(offset);
}

namespace bridge {
void trace_start(std::string_view id) {
  BridgeLogger::instance().log(id, true);
}
void trace_end(std::string_view id) {
  BridgeLogger::instance().log(id, false);
}
bool is_enabled() {
  return BridgeLogger::instance().enabled();
}
}; // namespace bridge
}; // namespace profile
}; // namespace habana