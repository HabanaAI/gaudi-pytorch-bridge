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
    std::vector<device_ptr>&& device_ptrs,
    stream& stream,
    event_done_callback done_cb) {
  auto eref = std::make_shared<event>(
      stream.get_device().get_event_handle_cache(),
      stream,
      std::move(device_ptrs),
      std::move(done_cb));
  PT_SYNHELPER_DEBUG("Adding new event ", *eref, " on stream ", stream);
  {
    std::unique_lock<std::mutex> lock(mut_);
    for (const auto& device_address : eref->get_device_ptrs()) {
      PT_SYNHELPER_DEBUG(
          "Adding producer for address ",
          std::hex,
          device_address,
          " on event ",
          *eref);
      auto found = events_.find(device_address);

      if (found != events_.end()) {
        shared_event event = found->second;
        lock.unlock();
        wait_until_done(event);
        lock.lock();
      }
      events_.emplace(device_address, eref);
    }
  }
  stream.register_pending_event(eref);
}

void stream_event_manager::enqueue_wait_event(
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
    event->stream_wait_event(stream);
  } else {
    PT_SYNHELPER_DEBUG("Event already done, as it's not in the map");
  }
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
    wait_until_done(evnt);
}

void stream_event_manager::wait_until_done(shared_event& event) {
  event->wait();
}

void stream_event_manager::synchronize_event(shared_event& event) {
  event->synchronize();
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    for (auto ptr : event->get_device_ptrs()) {
      auto it = events_.find(ptr);
      if (it == events_.end()) {
        PT_SYNHELPER_FATAL("cannot find event for address ", std::hex, ptr);
      }
      if (it->second != event) {
        PT_SYNHELPER_FATAL("pointer ", std::hex, ptr, " maps to another event");
      }
      PT_SYNHELPER_DEBUG("unmapping event for address ", std::hex, ptr);
      events_.erase(it);
    }
  }
  event->complete();
}

bool stream_event_manager::is_flushed() {
  std::lock_guard<std::mutex> lock_guard(mut_);
  return events_.empty();
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
