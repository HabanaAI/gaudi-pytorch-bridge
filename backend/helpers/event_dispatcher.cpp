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
#include "event_dispatcher.h"
#include <c10/util/Exception.h>
#include <chrono>
#include <memory>
#include <ostream>
#include "habana_helpers/logging.h"

namespace habana_helpers {

EventDispatcher::EventDispatcher() : next_subscribe_id_(0) {}

std::shared_ptr<EventDispatcherHandle> EventDispatcher::subscribe(
    EventDispatcher::Topic topic,
    const EventDispatcher::EventCallback& callback) {
  std::lock_guard<std::mutex> ld(mutex_);
  subscribers_[topic].push_back(
      {next_subscribe_id_++,
       std::make_shared<EventDispatcher::EventCallback>(callback)});

  return EventDispatcherHandle::create(topic, subscribers_[topic].back().first);
}

void EventDispatcher::publish(
    EventDispatcher::Topic topic,
    const EventDispatcher::EventParams& params,
    EventTsType timestamp) {
  log_publish_request(topic, params);

  std::vector<EventCallbackWithSubId> copy_of_subs;

  {
    std::lock_guard<std::mutex> ld(mutex_);
    if (subscribers_.count(topic) == 0) {
      return;
    } else {
      // Copy subscribers list to limit the scope of mutex being locked,
      // especially to avoid calling callback (usually Python code) with locked
      // mutex (mutex_). Calling Python callback with mutex_ taken may result in
      // deadlock with GIL, example:
      // 1. Publish is called by C++ thread, for example graph compilation
      //   - lock is taken by pure C++ thread
      //   - C++ thread is preempted
      // 2. Python main thread calls Python API that publishes event
      //   - GIL is taken by Python main thread
      //   - publish is called
      //   - Python main thread hangs on mutex_, because it is already taken by
      //     C++ thread
      // 3. C++ thread is resumed and tries to call Python callback
      //   - since C++ calls Python, then code tries to acquire GIL
      //   - GIL is already taken by Python main thread
      //   - nothing more can be done: C++ thread waits for GIL and Python
      //   thread
      //     is not able to return it, because it waits for mutex_ that is
      //     already taken by C++ thread
      copy_of_subs = subscribers_[topic];
    }
  }

  for (const auto& callback_with_sub_id : copy_of_subs) {
    (*callback_with_sub_id.second)(timestamp, params);
  }
}

void EventDispatcher::unsubscribe_all() {
  std::lock_guard<std::mutex> ld(mutex_);
  subscribers_.clear();
}

void EventDispatcher::unsubscribe(Topic topic, int64_t sub_id) {
  std::lock_guard<std::mutex> ld(mutex_);

  if (subscribers_.count(topic) == 0) {
    return;
  }

  auto new_end = std::remove_if(
      subscribers_[topic].begin(),
      subscribers_[topic].end(),
      [sub_id](auto& callback_with_id) {
        return callback_with_id.first == sub_id ? true : false;
      });

  // shrink callback list by one element when element was found
  if (new_end != subscribers_[topic].end()) {
    subscribers_[topic].pop_back();
  }

  TORCH_CHECK(
      new_end == subscribers_[topic].end(),
      "Inconsistent size of event dispatcher callback list");
}

void EventDispatcher::unsubscribe(
    const std::shared_ptr<EventDispatcherHandle>& handle) {
  unsubscribe(handle->topic, handle->sub_id);
}

void EventDispatcher::log_publish_request(
    EventDispatcher::Topic topic,
    const EventDispatcher::EventParams& params) {
  PT_HABHELPER_DEBUG(
      "Published topic: ", topic, " with ", params.size(), " parameters:");
  for (auto& entry : params) {
    auto param_name = entry.first;
    auto param_data = entry.second;
    PT_HABHELPER_DEBUG("param | [", param_name, "]=", param_data, " |");
  }
}
}; // namespace habana_helpers
