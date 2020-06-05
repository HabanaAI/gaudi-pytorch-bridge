/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/event_handle_cache.h"

#include <synapse_api.h>
#include <synapse_common_types.h>

#include "synapse_helpers/device.h"
#include "synapse_helpers/logging.h"
// IWYU pragma: no_include <ostream>

// There is a hard limit in Synapse for number of silmuntaneously recorded events on streams. Since, in TF, each event
// corresponds to single tensor, either being transfered or worked on, we can easily reach the point, where we have
// too many events used at once, hence the limit. In the future, to be on a safe side, we might consider creating
// bundles of tensors for single event, thus reducing overall number of events in use.
constexpr std::size_t MAX_NUM_EVENTS = 1000;

namespace synapse_helpers {
event_handle_cache::event_handle_cache(device& device) : mutex_{}, cond_var_{}, device_{device}, events_count_{0} {
  free_handles_.reserve(MAX_NUM_EVENTS);
}

synEventHandle event_handle_cache::get_free_handle() {
  synEventHandle handle;
  std::unique_lock<std::mutex> lock(mutex_);
  if (!free_handles_.empty()) {
    handle = free_handles_.back();
    free_handles_.pop_back();
    return handle;
  }

  // special case, if max number of events was reached,
  // we need to wait until an event is returned to the cache
  if (events_count_ >= MAX_NUM_EVENTS) {
    while (free_handles_.empty()) {
      cond_var_.wait(lock);
    }
    handle = free_handles_.back();
    free_handles_.pop_back();
    return handle;
  }

  auto status{synEventCreate(&handle, device_.id(), EVENT_FLAGS)};
  if (synStatus::synSuccess != status) {
    LOG_(FATAL) << "Event creation failed";
  } else {
    ++events_count_;
  }
  return handle;
}

void event_handle_cache::release_handle(synEventHandle handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!handle) {
    LOG_(FATAL) << "attempt to release null event handle";
  }
  free_handles_.push_back(handle);
  cond_var_.notify_one();
}

event_handle_cache::~event_handle_cache() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto handle : free_handles_) {
    synEventDestroy(handle);
  }
  free_handles_.clear();
}

}  // namespace synapse_helpers
