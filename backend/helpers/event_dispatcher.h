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

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "habana_helpers/logging.h"

namespace habana_helpers {

class EventDispatcherHandle;
class EventDispatcher {
 public:
  enum class Topic {
    GRAPH_COMPILE,
    MARK_STEP,
    PROCESS_EXIT,
    DEVICE_ACQUIRED,
    CUSTOM_EVENT,
    MEMORY_DEFRAGMENTATION,
    CPU_FALLBACK,
    CACHE_HIT,
    CACHE_MISS

  };

  static EventDispatcher& Instance() {
    static EventDispatcher instance;
    return instance;
  }

  using EventParam = std::pair<std::string, std::string>;
  using EventParams = std::vector<EventParam>;
  using EventTsType = std::chrono::time_point<std::chrono::system_clock>;
  using EventCallbackFuncType = void(EventTsType timestamp, const EventParams&);
  using EventCallback = std::function<EventCallbackFuncType>;
  using EventCallbackWithSubId =
      std::pair<int64_t, std::shared_ptr<EventCallback>>;

  std::shared_ptr<EventDispatcherHandle> subscribe(
      Topic topic,
      const EventCallback& callback);
  void unsubscribe(Topic topic, int64_t subscribe_id);
  void unsubscribe(const std::shared_ptr<EventDispatcherHandle>& handle);
  void publish(
      Topic topic,
      const EventParams& params,
      EventTsType timestamp = std::chrono::system_clock::now());
  void unsubscribe_all();

 private:
  EventDispatcher();
  std::unordered_map<Topic, std::vector<EventCallbackWithSubId>> subscribers_;
  std::mutex mutex_;
  int64_t next_subscribe_id_;

  void log_publish_request(Topic topic, const EventParams& params);
};

class EventDispatcherHandle
    : public std::enable_shared_from_this<EventDispatcherHandle> {
 public:
  const EventDispatcher::Topic topic;
  const int64_t sub_id;
  std::shared_ptr<EventDispatcherHandle> getptr() {
    return shared_from_this();
  }

  [[nodiscard]] static std::shared_ptr<EventDispatcherHandle> create(
      EventDispatcher::Topic topic,
      int64_t id) {
    return std::shared_ptr<EventDispatcherHandle>(
        new EventDispatcherHandle(topic, id));
  }

 private:
  EventDispatcherHandle(EventDispatcher::Topic topic, int64_t sub_id)
      : topic(topic), sub_id(sub_id) {}
};

inline void EmitEvent(
    EventDispatcher::Topic event_id,
    const EventDispatcher::EventParams& params =
        habana_helpers::EventDispatcher::EventParams(),
    EventDispatcher::EventTsType timestamp = std::chrono::system_clock::now()) {
  EventDispatcher::Instance().publish(event_id, params, timestamp);
}

inline std::ostream& operator<<(
    std::ostream& o,
    const EventDispatcher::Topic& t) {
  switch (t) {
    case EventDispatcher::Topic::GRAPH_COMPILE:
      o << "GRAPH_COMPILE";
      break;
    case EventDispatcher::Topic::MARK_STEP:
      o << "MARK_STEP";
      break;
    case EventDispatcher::Topic::PROCESS_EXIT:
      o << "PROCESS_EXIT";
      break;
    case EventDispatcher::Topic::DEVICE_ACQUIRED:
      o << "DEVICE_ACQUIRED";
      break;
    case EventDispatcher::Topic::CUSTOM_EVENT:
      o << "CUSTOM_EVENT";
      break;
    case EventDispatcher::Topic::MEMORY_DEFRAGMENTATION:
      o << "MEMORY_DEFRAGMENTATION";
      break;
    case EventDispatcher::Topic::CPU_FALLBACK:
      o << "CPU_FALLBACK";
      break;
    case EventDispatcher::Topic::CACHE_HIT:
      o << "CACHE_HIT";
      break;
    case EventDispatcher::Topic::CACHE_MISS:
      o << "CACHE_MISS";
      break;
  }
  return o;
}

}; // namespace habana_helpers
