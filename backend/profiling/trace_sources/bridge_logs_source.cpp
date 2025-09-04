/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "bridge_logs_source.h"
#include <syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_set>
#include "pytorch_helpers/habana_helpers/logging.h"

namespace {
uint64_t nowNanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count());
}
} // namespace

namespace habana::profile {

struct BridgeLogsSourceImpl : public TraceSource {
  BridgeLogsSourceImpl() = default;
  ~BridgeLogsSourceImpl() override = default;
  void log(std::string_view id, bool is_begin) {
    if (enabled(id)) {
      const auto dtime = nowNanos();
      const auto tid = static_cast<pid_t>(syscall(__NR_gettid));
      std::lock_guard<std::mutex> lg{m};
      updateThreadNames(tid);
      events_.emplace_back(std::string{id}, dtime, tid, is_begin);
    }
  }
  void set_mandatory_events(
      const std::vector<std::string>& mandatory_events,
      bool catch_all_events) {
    std::copy(
        std::begin(mandatory_events),
        std::end(mandatory_events),
        std::inserter(mandatory_events_, mandatory_events_.end()));
    mandatory_list_initialized_ = true;
    catch_all_events_ = catch_all_events;
  }
  bool enabled(std::string_view name = "") {
    if (is_started_) {
      if (catch_all_events_) {
        return true;
      }
      if (mandatory_list_initialized_) {
        return exists_on_mandatory_list(name);
      }
    }
    return false;
  }
  bool exists_on_mandatory_list(std::string_view name = "") {
    // Below memoization technique is used to cache already computed values for
    // particular functions. Each function name is validated using regex rules
    // stored inside mandatory_events_. This computation could be expensive so
    // results are stored.
    {
      std::lock_guard<std::mutex> lg{checked_.m};
      auto it_checked = checked_.go.find(name.data());
      if (it_checked != checked_.go.end()) {
        return it_checked->second;
      }
    }
    bool matched{false};
    for (const auto& mandatory_event : mandatory_events_) {
      std::regex mandatory_event_regex(
          mandatory_event, std::regex_constants::ECMAScript);
      if (std::regex_search(name.begin(), name.end(), mandatory_event_regex)) {
        matched = true;
        break;
      }
    }
    std::lock_guard<std::mutex> lg{checked_.m};
    checked_.go.emplace(name.data(), matched);
    return matched;
  }
  static BridgeLogsSourceImpl& instance() {
    static BridgeLogsSourceImpl source;
    return source;
  }
  void start(TraceSink&) override {
    is_started_ = true;
  }
  void stop() override {
    is_started_ = false;
  }
  void extract(TraceSink& output) override {
    if (events_.empty())
      return;
    pid_t pid = static_cast<pid_t>(getpid()) + offset_;
    std::lock_guard<std::mutex> lg{m};
    for (const auto& event : events_) {
      output.addActivity(
          {event.name, {}, ActivityType::RUNTIME, pid, event.tid},
          {},
          event.time,
          event.begin);
    }
    for (const auto& entry : threadNames) {
      std::string name =
          "thread " + std::to_string(entry.first) + " (" + entry.second + ")";
      output.addResource(name, pid, entry.first);
    }
    events_.clear();
  }
  TraceSourceVariant get_variant() override {
    return TraceSourceVariant::BRIDGE_LOGS;
  }
  void set_offset(unsigned offset) override {
    offset_ = offset;
  }

 private:
  void updateThreadNames(pid_t tid) {
    if (not threadNames.count(tid)) {
      auto name = getThreadName();
      threadNames[tid] = name;
    }
  }
  struct Event {
    std::string name;
    uint64_t time;
    pid_t tid;
    bool begin;
    Event(std::string&& name, uint64_t time, pid_t tid, bool begin)
        : name(std::move(name)), time(time), tid(tid), begin(begin) {}
  };
  std::deque<Event> events_;
  std::atomic<bool> is_started_{false};
  std::atomic<bool> mandatory_list_initialized_{false};
  std::atomic<bool> catch_all_events_{false};
  std::unordered_set<std::string> mandatory_events_;
  struct {
    std::unordered_map<const char*, bool> go;
    std::mutex m{};
  } checked_;
  unsigned offset_{};
  std::mutex m{};
  std::unordered_map<int64_t, std::string> threadNames;
};

BridgeLogsSource::BridgeLogsSource(
    bool is_requested,
    const std::vector<std::string>& mandatory_events) {
  BridgeLogsSourceImpl::instance().set_mandatory_events(
      mandatory_events, is_requested);
}

BridgeLogsSource::~BridgeLogsSource() {}

void BridgeLogsSource::start(TraceSink& sink) {
  BridgeLogsSourceImpl::instance().start(sink);
}
void BridgeLogsSource::stop() {
  BridgeLogsSourceImpl::instance().stop();
}
void BridgeLogsSource::extract(TraceSink& output) {
  BridgeLogsSourceImpl::instance().extract(output);
}

TraceSourceVariant BridgeLogsSource::get_variant() {
  return BridgeLogsSourceImpl::instance().get_variant();
}
void BridgeLogsSource::set_offset(unsigned offset) {
  BridgeLogsSourceImpl::instance().set_offset(offset);
}

namespace bridge {
void trace_start(std::string_view id) {
  BridgeLogsSourceImpl::instance().log(id, true);
}
void trace_end(std::string_view id) {
  BridgeLogsSourceImpl::instance().log(id, false);
}
bool is_enabled(std::string_view name) {
  return BridgeLogsSourceImpl::instance().enabled(name);
}
}; // namespace bridge
}; // namespace habana::profile
