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
#include <synapse_api_types.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace synapse_helpers {
class device;

class event_handle_cache {
 public:
  explicit event_handle_cache(device& device, uint32_t event_flag);
  event_handle_cache(const event_handle_cache&) = delete;
  event_handle_cache(event_handle_cache&&) = delete;
  event_handle_cache& operator=(const event_handle_cache&) = delete;
  event_handle_cache& operator=(event_handle_cache&&) = delete;

  ~event_handle_cache();
  synEventHandle get_free_handle();
  size_t get_free_events_count() {
    return free_handles_.size();
  }
  size_t get_total_events_count() {
    return events_count_;
  }
  void release_handle(synEventHandle handle);
  static size_t get_num_events_high_watermark() {
    return NUM_EVENTS_HIGH_WATERMARK;
  }

 private:
  const uint32_t event_flag_ = 0;

  std::vector<synEventHandle> free_handles_;
  std::mutex mutex_;
  std::condition_variable cond_var_;
  device& device_;

  // value to control the total number of synEventHandles created
  static size_t events_count_;

  // There is a hard limit in Synapse for number of silmuntaneously recorded
  // events on streams. Since, in TF, each event corresponds to single tensor,
  // either being transfered or worked on, we can easily reach the point, where
  // we have too many events used at once, hence the limit. In the future, to be
  // on a safe side, we might consider creating bundles of tensors for single
  // event, thus reducing overall number of events in use.
  static constexpr size_t NUM_EVENTS_MAX = 1000000;
  static constexpr size_t NUM_EVENTS_HIGH_WATERMARK = NUM_EVENTS_MAX - 100;

  // when event max has reached, check if any recipe exeution completion
  // will help in freeing the event. recipe_count in queue must be greater
  // than 1 as recipe is incremented after launching and before the adding
  // of producer events
  static constexpr size_t NUM_RECIPE_COUNT_TO_WAIT_FOR_FREE_EVENT = 1;
};

class CachedEventHandle {
 public:
  CachedEventHandle(event_handle_cache& event_cache)
      : event_handle_cache_(event_cache),
        event_(event_handle_cache_.get_free_handle()) {}
  ~CachedEventHandle() {
    if (active)
      event_handle_cache_.release_handle(event_);
  }
  synEventHandle get() const {
    return event_;
  };

  CachedEventHandle(const CachedEventHandle&) = delete;
  CachedEventHandle& operator=(const CachedEventHandle&) = delete;
  CachedEventHandle(CachedEventHandle&& other) noexcept
      : active(other.active),
        event_handle_cache_(other.event_handle_cache_),
        event_(other.event_) {
    other.active = false;
  }
  CachedEventHandle& operator=(CachedEventHandle&&) = delete;

 private:
  bool active = true;
  event_handle_cache& event_handle_cache_;
  synEventHandle event_;
};

} // namespace synapse_helpers
