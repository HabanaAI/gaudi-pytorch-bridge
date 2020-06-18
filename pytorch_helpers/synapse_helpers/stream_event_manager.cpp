/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/stream_event_manager.h"

#include <synapse_common_types.h>

#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "habana_helpers/logging.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/stream.h"

using namespace synapse_helpers;

void stream_event_manager::add_producer(
    const std::vector<device_ptr>& device_ptrs,
    stream& stream,
    event_done_callback done_cb) {
  auto eref = std::make_shared<event>(
      stream.get_device().get_event_handle_cache(), stream, std::move(done_cb));
  PT_SYNHELPER_DEBUG("Adding new event ", *eref, " on stream ", stream);
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    for (const auto& device_address : device_ptrs) {
      PT_SYNHELPER_DEBUG(
          "Adding producer for address ",
          std::hex,
          device_address,
          " on event ",
          *eref);
      auto found = events_.find(device_address);

      if (found != events_.end()) { // NOLINT
        if (found->second != nullptr && !found->second->done()) { // NOLINT
          if (synStatus::synSuccess != found->second->synchronize())
            PT_SYNHELPER_FATAL(
                "Failed to synchronize event in order to register new producer to same tensor address (0x",
                std::hex,
                device_address,
                ")");
        }
        found->second = eref; // NOLINT
      } else {
        events_.emplace(device_address, eref);
      }
    }
  }
  stream.register_pending_event(eref);
}

bool stream_event_manager::record_wait_event(
    device_ptr device_address,
    stream& stream) {
  PT_SYNHELPER_DEBUG(
      "Recording wait event on stream ",
      stream,
      " for device address ",
      std::hex,
      device_address);
  shared_event event;
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    auto it = events_.find(device_address);
    if (it != events_.end()) {
      event = it->second;
    }
  }
  if (event) {
    PT_SYNHELPER_DEBUG(
        "Found event ", *event, " for address ", std::hex, device_address);
    if (event->streamWaitEvent(stream)) {
      return true;
    } else {
      PT_SYNHELPER_DEBUG(
          "Event ", *event, " is already done. No wait event was recorded.");
      // event is already done
      std::lock_guard<std::mutex> lock_guard(mut_);
      events_.erase(device_address);
      return false;
    }
  }
  // else already done, as it's not in the map
  return false;
}

void stream_event_manager::wait_until_done(device_ptr device_address) {
  shared_event evnt{};
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    auto it = events_.find(device_address);
    if (it != events_.end()) {
      evnt = it->second;
    } else {
      return;
    }
  }

  if (evnt)
    evnt->synchronize();

  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    events_.erase(device_address);
  }
}

void stream_event_manager::flush() {
  bool events_not_empty = true;
  while (events_not_empty) {
    clear_if_done();
    {
      std::lock_guard<std::mutex> lock_guard(mut_);
      events_not_empty = !events_.empty();
    }
  }
}

shared_event stream_event_manager::get_event(device_ptr device_address) {
  std::lock_guard<std::mutex> lock_guard(mut_);
  auto it = events_.find(device_address);
  if (it != events_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

void stream_event_manager::clear_if_done() {
  std::vector<shared_event> done_events;
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    done_events.reserve(events_.size());
    for (auto it = events_.begin(); it != events_.end();) {
      auto copy_it = it++;
      if (copy_it->second->done()) {
        // Events to delete are moved to separate container in order to
        // be removed outside of lock_guard scope.
        done_events.push_back(std::move(copy_it->second));
        events_.erase(copy_it);
      }
    }
  }
}
