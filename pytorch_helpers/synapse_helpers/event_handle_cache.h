/******************************************************************************
 * Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
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
  void release_handle(synEventHandle handle);

 private:
  const uint32_t event_flag_ = 0;

  std::vector<synEventHandle> free_handles_;
  std::mutex mutex_;
  std::condition_variable cond_var_;
  device& device_;
  std::size_t events_count_; // value to control the total number of
                             // synEventHandles created
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
