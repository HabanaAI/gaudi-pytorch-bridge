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
#include "backend/synapse_helpers/stream_event_manager.h"

#include <synapse_common_types.h>

#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/stream.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/python_utils.h"

using namespace synapse_helpers;

void stream_event_manager::add_producer(
    std::vector<device_ptr>&& device_ptrs,
    stream& stream,
    event_done_callback done_cb) {
  add_producer(std::move(device_ptrs), "", stream, std::move(done_cb));
}

void stream_event_manager::add_future(
    device_ptr device_address,
    std::shared_future<bool> fut) {
  PT_SYNHELPER_TRACE;
  std::lock_guard<std::mutex> lock(future_mut_);
  auto found = future_by_addr_.find(device_address);
  if (found != future_by_addr_.end()) {
    HABANA_ASSERT(found->second.valid());
    // Release GIL if going to wait. This thread might already acquired GIL and
    // the second thread will be waiting
    habana_helpers::AutoNoGIL gil_release;
    found->second.wait();
  }
  future_by_addr_[device_address] = std::move(fut);
}

void stream_event_manager::wait_for_future(device_ptr device_address) {
  PT_SYNHELPER_TRACE;
  std::shared_future<bool> fut;
  {
    std::lock_guard<std::mutex> lock(future_mut_);
    auto it = future_by_addr_.find(device_address);
    if (it == future_by_addr_.end()) {
      return;
    }
    fut = it->second;
  }
  HABANA_ASSERT(fut.valid());
  // Release GIL if going to wait. This thread might already acquired GIL and
  // the second thread will be waiting
  {
    habana_helpers::AutoNoGIL gil_release;
    fut.wait();
  }
  {
    std::lock_guard<std::mutex> lock(future_mut_);
    future_by_addr_.erase(device_address);
  }
}

void stream_event_manager::wait_for_all_futures() {
  PT_SYNHELPER_TRACE;
  std::vector<device_ptr> device_addresses;
  {
    std::lock_guard<std::mutex> lock(future_mut_);
    for (auto& fut : future_by_addr_) {
      device_addresses.emplace_back(fut.first);
    }
  }
  for (auto& dev_addr : device_addresses) {
    wait_for_future(dev_addr);
  }
}

void stream_event_manager::add_producer(
    std::vector<device_ptr>&& device_addresses,
    std::string event_id,
    stream& stream,
    event_done_callback done_cb) {
  // remove duplicate address
  std::sort(device_addresses.begin(), device_addresses.end());
  device_addresses.erase(
      std::unique(device_addresses.begin(), device_addresses.end()),
      device_addresses.end());
  shared_event eref;
  eref = std::make_shared<event>(
      stream.get_device().get_event_handle_cache(),
      stream,
      std::move(device_addresses),
      std::string{event_id},
      std::move(done_cb));
  add_producer(event_id, stream, eref);
}

void stream_event_manager::add_producer(
    std::string event_id,
    stream& stream,
    shared_event eref) {
  PT_SYNHELPER_DEBUG("Adding new event ", *eref, " on stream ", stream);
  {
    std::lock_guard<std::mutex> lock(mut_);
    for (const auto& device_address : eref->get_device_ptrs()) {
      PT_SYNHELPER_DEBUG(
          "Adding producer for address ",
          reinterpret_cast<void*>(device_address),
          " on event ",
          *eref);
      auto found = events_by_addr_.find(device_address);

      if (found != events_by_addr_.end()) {
        // Address collision on this point actually means that we already
        // scheduled work that will override data associated with old event.
        // This should only happen in case when output and input buffers of
        // operation are the same, and there is noone else waiting for previous
        // event. In that case we do not want to wait for event to synchronize
        // as this will postpone launching next ops in graph.
        PT_SYNHELPER_DEBUG(
            "Event collision on address: 0x",
            reinterpret_cast<void*>(device_address));
        shared_event event = found->second;
        event->remove_device_ptr(device_address);
        events_by_addr_.erase(found);
      }
      events_by_addr_.emplace(device_address, eref);
    }
    auto id_found = events_by_str_.find(event_id);
    if (id_found != events_by_str_.end()) {
      // we should have already found it by address and by now it would be gone
      // since we called wait_until_done
      PT_SYNHELPER_FATAL(
          "events by address and by string are out of sync for ", event_id);
    }
    if (!event_id.empty()) {
      events_by_str_.emplace(event_id, eref);
    }
  }
  stream.register_pending_event(eref);
}
void stream_event_manager::add_producer(stream& stream, shared_event event) {
  const auto& device_addresses = event->get_device_ptrs();
  for (auto& address : device_addresses) {
    auto found = events_by_addr_.find(address);
    if (found == events_by_addr_.end()) {
      PT_SYNHELPER_FATAL(
          "Cannot find event registed for address: 0x",
          reinterpret_cast<void*>(address));
    }
    if (found->second != event) {
      PT_SYNHELPER_FATAL(
          "Event does not match for address: 0x",
          reinterpret_cast<void*>(address));
    }
  }
  stream.register_pending_event(event, false);
}

void stream_event_manager::add_event_id(
    const std::string& event_id,
    const std::string& new_id) {
  std::lock_guard<std::mutex> lock_guard(mut_);
  auto it = events_by_str_.find(event_id);
  if (it == events_by_str_.end()) {
    return;
  }
  PT_SYNHELPER_DEBUG("Found event ", *(it->second), " for id ", event_id);
  it->second->push_id(new_id);
  events_by_str_.insert(std::make_pair(new_id, it->second));
}

shared_event stream_event_manager::map_event_to_tensor(
    stream& stream,
    const synRecipeHandle recipe_handle,
    synLaunchTensorInfo* tensor_info,
    event_done_callback done_cb) {
  std::lock_guard<std::mutex> lock(mut_);
  auto& device_address = tensor_info->pTensorAddress;
  auto found = events_by_addr_.find(device_address);
  if (found != events_by_addr_.end()) {
    // Address collision on this point actually means that we already scheduled
    // work that will override data associated with old event. This should only
    // happen in case when output and input buffers of operation are the same,
    // and there is noone else waiting for previous event. In that case we do
    // not want to wait for event to synchronize as this will postpone launching
    // next ops in graph.
    PT_SYNHELPER_DEBUG(
        "Event collision on address: 0x",
        reinterpret_cast<void*>(device_address));

    auto found_event = found->second;
    found_event->remove_device_ptr(device_address);
    events_by_addr_.erase(device_address);
  }

  auto shared_event = std::make_shared<event>(
      stream.get_device().get_event_handle_cache(),
      stream,
      std::vector<device_ptr>{device_address},
      std::string{},
      std::move(done_cb));
  shared_event->map_event_to_tensor(recipe_handle, tensor_info);
  PT_SYNHELPER_DEBUG(
      "External event ",
      *shared_event,
      " mapped to address ",
      reinterpret_cast<void*>(device_address));
  events_by_addr_.emplace(device_address, shared_event);

  return shared_event;
}

void stream_event_manager::enqueue_wait_event(
    device_ptr device_address,
    stream& stream) {
  PT_SYNHELPER_DEBUG(
      "stream ",
      stream,
      " waits for event mapped to device address ",
      reinterpret_cast<void*>(device_address));

  wait_for_future(device_address);
  std::lock_guard<std::mutex> lock_guard(mut_);
  shared_event event;
  {
    auto it = events_by_addr_.find(device_address);
    if (it != events_by_addr_.end()) {
      event = it->second;
    }
  }
  if (event) {
    PT_SYNHELPER_DEBUG(
        "Found event ",
        *event,
        " for address ",
        reinterpret_cast<void*>(device_address));
    event->stream_wait_event(stream);
  } else {
    PT_SYNHELPER_DEBUG("Event already done, as it's not in the map");
  }
}

shared_event stream_event_manager::get_wait_event(
    device_ptr device_address,
    stream& stream) {
  PT_SYNHELPER_DEBUG(
      "stream ",
      stream,
      " waits for event mapped to device address ",
      reinterpret_cast<void*>(device_address));

  wait_for_future(device_address);
  std::lock_guard<std::mutex> lock_guard(mut_);
  shared_event event;
  {
    auto it = events_by_addr_.find(device_address);
    if (it != events_by_addr_.end()) {
      event = it->second;
    }
  }
  return event;
}

void stream_event_manager::enqueue_wait_event(
    const std::string& event_id,
    stream& stream) {
  PT_SYNHELPER_DEBUG(
      "stream ", stream, " waits for event mapped to id ", event_id);

  std::lock_guard<std::mutex> lock_guard(mut_);
  shared_event event;
  {
    auto it = events_by_str_.find(event_id);
    if (it != events_by_str_.end()) {
      event = it->second;
    }
  }
  if (event) {
    PT_SYNHELPER_DEBUG("Found event ", *event, " for id ", event_id);
    event->stream_wait_event(stream);
  } else {
    PT_SYNHELPER_DEBUG("Event already done, as it's not in the map");
  }
}

void stream_event_manager::wait_until_done(device_ptr device_address) {
  wait_for_future(device_address);
  shared_event evnt{};
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    auto it = events_by_addr_.find(device_address);
    if (it != events_by_addr_.end()) {
      evnt = it->second;
    } else {
      return;
    }
  }

  if (evnt)
    wait_until_done(evnt);
}

void stream_event_manager::wait_until_done(const std::string& event_id) {
  shared_event evnt{};
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    auto it = events_by_str_.find(event_id);
    if (it != events_by_str_.end()) {
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
  // The sync_mut_ is to be held till the call back tensors are released
  event->synchronize();
  std::lock_guard<std::mutex> sync_mut_lock_guard(sync_mut_);
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    for (auto ptr : event->get_device_ptrs()) {
      auto it = events_by_addr_.find(ptr);
      if (it == events_by_addr_.end()) {
        PT_SYNHELPER_FATAL(
            "cannot find event for address ", reinterpret_cast<void*>(ptr));
      }
      if (it->second != event) {
        PT_SYNHELPER_FATAL(
            "pointer ", reinterpret_cast<void*>(ptr), " maps to another event");
      }
      PT_SYNHELPER_DEBUG(
          "unmapping event for address ", reinterpret_cast<void*>(ptr));
      events_by_addr_.erase(it);
    }
    for (auto& event_id : event->get_event_ids()) {
      auto it = events_by_str_.find(event_id);
      if (it == events_by_str_.end()) {
        PT_SYNHELPER_FATAL("cannot find event for event id \"", event_id, "\"");
      }
      if (it->second != event) {
        PT_SYNHELPER_FATAL("event id ", event_id, " maps to another event");
      }
      PT_SYNHELPER_DEBUG("unmapping event for id ", event_id);

      events_by_str_.erase(it);
    }
  }
  event->complete();
}

bool stream_event_manager::is_flushed() {
  std::lock_guard<std::mutex> sync_mut_lock_guard(sync_mut_);
  std::lock_guard<std::mutex> lock_guard(mut_);
  return events_by_addr_.empty() && events_by_str_.empty();
}

shared_event stream_event_manager::get_event(device_ptr device_address) {
  std::lock_guard<std::mutex> lock_guard(mut_);
  auto it = events_by_addr_.find(device_address);
  if (it != events_by_addr_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}
