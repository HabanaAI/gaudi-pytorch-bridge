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
#include "event_dispatcher.h"
#include <c10/util/Exception.h>
#include <chrono>
#include <ostream>
#include "habana_helpers/logging.h"

namespace habana_helpers {

EventDispatcher::EventDispatcher() : next_subscribe_id_(0) {}

std::shared_ptr<EventDispatcherHandle> EventDispatcher::subscribe(
    EventDispatcher::Topic topic,
    const EventDispatcher::EventCallback& callback) {
  std::lock_guard<std::mutex> ld(mutex_);
  subscribers_[topic].push_back({next_subscribe_id_++, callback});

  return EventDispatcherHandle::create(topic, subscribers_[topic].back().first);
}

void EventDispatcher::publish(
    EventDispatcher::Topic topic,
    const EventDispatcher::EventParams& params,
    EventTsType timestamp) {
  log_publish_request(topic, params);

  std::lock_guard<std::mutex> ld(mutex_);
  if (subscribers_.count(topic) == 0) {
    return;
  }

  for (const auto& callback_with_sub_id : subscribers_[topic]) {
    callback_with_sub_id.second(timestamp, params);
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
    std::visit(
        [param_name](auto&& param_data_unpacked) {
          PT_HABHELPER_DEBUG(
              "param | [", param_name, "]=", param_data_unpacked, " | ");
        },
        param_data);
  }
}
}; // namespace habana_helpers