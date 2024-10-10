/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
