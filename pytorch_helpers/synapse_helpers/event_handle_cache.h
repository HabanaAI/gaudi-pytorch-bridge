/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
  explicit event_handle_cache(device& device);
  event_handle_cache(const event_handle_cache&) = delete;
  event_handle_cache(event_handle_cache&&) = delete;
  event_handle_cache& operator=(const event_handle_cache&) = delete;
  event_handle_cache& operator=(event_handle_cache&&) = delete;

  ~event_handle_cache();
  synEventHandle get_free_handle();
  void release_handle(synEventHandle handle);

 private:
  static const uint32_t EVENT_FLAGS = 0;

  std::vector<synEventHandle> free_handles_;
  std::mutex mutex_;
  std::condition_variable cond_var_;
  device& device_;
  std::size_t events_count_;  // value to control the total number of synEventHandles created
};

}  // namespace synapse_helpers
