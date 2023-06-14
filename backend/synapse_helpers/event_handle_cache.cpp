/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/synapse_helpers/event_handle_cache.h"

#include <synapse_api.h>
#include <synapse_common_types.h>

#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"
// IWYU pragma: no_include <ostream>

namespace synapse_helpers {
size_t event_handle_cache::events_count_{0};

event_handle_cache::event_handle_cache(device& device, uint32_t event_flag)
    : event_flag_(event_flag), mutex_{}, cond_var_{}, device_{device} {
  free_handles_.reserve(NUM_EVENTS_MAX);
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
  if (events_count_ >= NUM_EVENTS_MAX) {
    if (event_flag_) {
      PT_SYNHELPER_FATAL(
          "Reached Max No of Timer Events allowed, total events::",
          events_count_);
    }
    // NUM_EVENTS_MAX limited to 1000000, inline to hard limit in synapse.
    // Assert if no recipe is being executed.
    auto& recipe_counter = device_.get_active_recipe_counter();
    HABANA_ASSERT(
        recipe_counter.get_count() > NUM_RECIPE_COUNT_TO_WAIT_FOR_FREE_EVENT,
        "Event handle out of resources. Event count exceeds max allowed limit.");

    while (free_handles_.empty()) {
      cond_var_.wait(lock);
    }
    handle = free_handles_.back();
    free_handles_.pop_back();
    return handle;
  }

  auto status{synEventCreate(&handle, device_.id(), event_flag_)};
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "Event creation failed");
  } else {
    ++events_count_;
  }
  return handle;
}

void event_handle_cache::release_handle(synEventHandle handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!handle) {
    PT_SYNHELPER_FATAL("attempt to release null event handle");
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

} // namespace synapse_helpers
