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
#include <charconv>
#include <chrono>
#include <deque>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
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
struct EventHasher {
  std::size_t operator()(const std::pair<pid_t, int64_t>& p) const noexcept {
    std::size_t h1 = std::hash<pid_t>{}(p.first);
    std::size_t h2 = std::hash<int64_t>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};
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
  void log(std::string_view id, bool is_begin, size_t index) {
    if (enabled(id)) {
      const auto dtime = nowNanos();
      const auto tid = static_cast<pid_t>(syscall(__NR_gettid));
      std::string event_id{id};
      std::lock_guard<std::mutex> lg{m};
      updateThreadNames(tid);
      events_.emplace_back(std::move(event_id), dtime, tid, is_begin);
      eventsToIndexes_[{tid, dtime}] = index;
    }
  }

  size_t generateDebugIndex() {
    static std::atomic<size_t> debug_index{0};
    return debug_index.fetch_add(1, std::memory_order_relaxed);
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
  void start(TraceSink& /*output*/) override {
    is_started_ = true;
  }
  void stop() override {
    is_started_ = false;
  }
  void extract(TraceSink& output) override {
    if (events_.empty())
      return;
    const auto pid =
        static_cast<pid_t>(static_cast<unsigned int>(getpid()) + offset_);
    std::lock_guard<std::mutex> lg{m};
    for (const auto& event : events_) {
      if (eventsToIndexes_.count({event.tid, event.time})) {
        auto index = eventsToIndexes_[{event.tid, event.time}];
        output.addActivity(
            {event.name,
             {{"index", std::to_string(index)}},
             ActivityType::HPU_RUNTIME,
             pid,
             event.tid},
            {},
            event.time,
            event.begin);
      } else
        output.addActivity(
            {event.name, {}, ActivityType::HPU_RUNTIME, pid, event.tid},
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
    eventsToIndexes_.clear();
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
  std::unordered_map<std::pair<pid_t, int64_t>, size_t, EventHasher>
      eventsToIndexes_;
  std::deque<Event> events_;
  std::atomic<bool> is_started_{false};
  std::atomic<bool> mandatory_list_initialized_{false};
  std::atomic<bool> catch_all_events_{false};
  std::unordered_set<std::string> mandatory_events_;
  struct {
    std::unordered_map<const char*, bool> go;
    std::mutex m;
  } checked_;
  unsigned offset_{};
  std::mutex m;
  std::unordered_map<int64_t, std::string> threadNames;
};

BridgeLogsSource::BridgeLogsSource(
    bool is_requested,
    const std::vector<std::string>& mandatory_events) {
  BridgeLogsSourceImpl::instance().set_mandatory_events(
      mandatory_events, is_requested);
}

BridgeLogsSource::~BridgeLogsSource() = default;

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

std::unordered_map<std::string, std::size_t> RecipeRegistry::recipes_{};
std::shared_mutex RecipeRegistry::mtx_{};

size_t RecipeRegistry::invalidId() {
  return kInvalidDebugId;
}

void RecipeRegistry::registerRecipe(const std::string& name, std::size_t id) {
  if (!habana::profile::bridge::linked_events_enabled() or id == invalidId())
    return;

  std::unique_lock<std::shared_mutex> lock(mtx_);
  recipes_[name] = id;
}

size_t RecipeRegistry::getRecipeId(std::string_view name) {
  if (!habana::profile::bridge::linked_events_enabled())
    return invalidId();

  std::shared_lock<std::shared_mutex> lock(mtx_);
  auto it = recipes_.find(std::string(name));
  return (it != recipes_.end()) ? it->second : invalidId();
}

bool RecipeRegistry::hasRecipeName(std::string_view name) {
  if (!habana::profile::bridge::linked_events_enabled())
    return false;

  std::shared_lock<std::shared_mutex> lock(mtx_);
  return recipes_.find(std::string(name)) != recipes_.end();
}

namespace bridge {
bool linked_events_enabled() {
  return GET_ENV_FLAG_NEW(PT_PROFILER_EAGER_LINKED_EVENTS);
}
void trace_start(std::string_view id, size_t index) {
  if (!linked_events_enabled() or
      index == habana::profile::RecipeRegistry::invalidId()) {
    BridgeLogsSourceImpl::instance().log(id, true);
    return;
  }
  BridgeLogsSourceImpl::instance().log(id, true, index);
}
void trace_start(std::string_view id) {
  BridgeLogsSourceImpl::instance().log(id, true);
}
void trace_end(std::string_view id) {
  BridgeLogsSourceImpl::instance().log(id, false);
}
bool is_enabled(std::string_view name) {
  return BridgeLogsSourceImpl::instance().enabled(name);
}
size_t get_debug_index() {
  return BridgeLogsSourceImpl::instance().generateDebugIndex();
}
}; // namespace bridge
}; // namespace habana::profile
