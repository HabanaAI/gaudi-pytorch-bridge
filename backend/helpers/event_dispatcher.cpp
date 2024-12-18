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

std::pair<uint64_t, uint64_t> Subscribers::UpdateIndexes(
    int64_t sub_id,
    uint64_t event_size) {
  auto it = std::find_if(
      subscribers_.begin(), subscribers_.end(), [sub_id](auto& el) {
        return sub_id == el.index_;
      });
  HABANA_ASSERT(it != subscribers_.end(), "Subscriber not found");

  auto start_index = it->pos_;

  if (start_index == event_size)
    return {start_index, 0};

  it->pos_ = event_size;

  auto min_result = std::min_element(
      subscribers_.begin(), subscribers_.end(), [](auto& a, auto& b) {
        return a.pos_ < b.pos_;
      });
  auto min_index = min_result->pos_;

  for (auto& el : subscribers_)
    el.pos_ -= min_index;

  return {start_index, min_index};
}

void Subscribers::Insert(int64_t sub_id, uint64_t pos) {
  subscribers_.emplace_back(sub_id, pos);
}

void Subscribers::Delete(int64_t sub_id) {
  auto it = std::remove_if(
      subscribers_.begin(), subscribers_.end(), [sub_id](auto& el) {
        return el.index_ == sub_id;
      });
  HABANA_ASSERT(it != subscribers_.end());
  subscribers_.erase(it, subscribers_.end());
}

void Subscribers::DecreasePos() {
  for (auto& el : subscribers_) {
    if (el.pos_ > 0)
      --el.pos_;
  }
}

void TopicQueue::Process(int64_t sub_id, const EventCallback& precess_func) {
  auto [index, free_el] = subscribers_.UpdateIndexes(sub_id, events_.size());
  HABANA_ASSERT(index <= events_.size());
  for (size_t i = index; i < events_.size(); i++) {
    precess_func(events_[i]);
  }
  if (free_el > 0)
    events_.erase(events_.begin(), events_.begin() + free_el);
}

void TopicQueue::Reset(int64_t sub_id) {
  auto [index, free_el] = subscribers_.UpdateIndexes(sub_id, events_.size());
  HABANA_ASSERT(index <= events_.size());
  if (free_el > 0)
    events_.erase(events_.begin(), events_.begin() + free_el);
}

void TopicQueue::AddSubscriber(int64_t sub_id) {
  subscribers_.Insert(sub_id, events_.size());
}
void TopicQueue::RemoveSubscriber(int64_t sub_id) {
  Reset(sub_id);
  subscribers_.Delete(sub_id);
}
void TopicQueue::PushEvent(const EventParams& params) {
  if (events_.size() > max_events_) {
    events_.pop_front();
    subscribers_.DecreasePos();
  }
  events_.push_back(params);
}

EventDispatcher::EventDispatcher() : next_subscribe_id_(0) {}

void EventDispatcher::process(
    const std::shared_ptr<EventDispatcherHandle>& handle,
    const EventCallback& callback) {
  std::lock_guard<std::mutex> ld(mutex_);
  auto it = topic_queues_.find(handle->topic);
  HABANA_ASSERT(it != topic_queues_.end(), "Topic not found");
  it->second.Process(handle->sub_id, callback);
}

void EventDispatcher::reset(
    const std::shared_ptr<EventDispatcherHandle>& handle) {
  std::lock_guard<std::mutex> ld(mutex_);
  auto it = topic_queues_.find(handle->topic);
  HABANA_ASSERT(it != topic_queues_.end(), "Topic not found");
  it->second.Reset(handle->sub_id);
}

std::shared_ptr<EventDispatcherHandle> EventDispatcher::subscribe(
    EventDispatcher::Topic topic) {
  std::lock_guard<std::mutex> ld(mutex_);
  auto subscribe_id_ = next_subscribe_id_++;
  topic_queues_[topic].AddSubscriber(subscribe_id_);
  return EventDispatcherHandle::create(topic, subscribe_id_);
}

void EventDispatcher::publish(
    EventDispatcher::Topic topic,
    const EventParams& params) {
  log_publish_request(topic, params);

  auto it = topic_queues_.find(topic);
  if (it == topic_queues_.end()) {
    return;
  }
  std::lock_guard<std::mutex> ld(mutex_);
  it->second.PushEvent(params);
}

void EventDispatcher::unsubscribe_all() {
  std::lock_guard<std::mutex> ld(mutex_);
  topic_queues_.clear();
}

void EventDispatcher::unsubscribe(Topic topic, int64_t sub_id) {
  std::lock_guard<std::mutex> ld(mutex_);
  auto it = topic_queues_.find(topic);
  if (it == topic_queues_.end())
    return;

  it->second.RemoveSubscriber(sub_id);
}

void EventDispatcher::unsubscribe(
    const std::shared_ptr<EventDispatcherHandle>& handle) {
  unsubscribe(handle->topic, handle->sub_id);
}

void EventDispatcher::log_publish_request(
    EventDispatcher::Topic topic,
    const EventParams& params) {
  PT_HABHELPER_DEBUG(
      "Published topic: ", topic, " with ", params.size(), " parameters:");
  for (auto& entry : params) {
    auto param_name = entry.first;
    auto param_data = entry.second;
    PT_HABHELPER_DEBUG("param | [", param_name, "]=", param_data, " |");
  }
}
}; // namespace habana_helpers
