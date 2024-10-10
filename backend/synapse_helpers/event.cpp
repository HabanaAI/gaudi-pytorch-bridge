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
#include "backend/synapse_helpers/event.h"

#include <synapse_api.h>

#include <ostream>
#include <utility>

#include "backend/synapse_helpers/event_handle_cache.h"
#include "backend/synapse_helpers/stream.h"
#include "habana_helpers/logging.h"

using namespace synapse_helpers;

event::event(
    event_handle_cache& event_handle_cache,
    stream& stream,
    std::vector<device_ptr>&& device_ptrs,
    std::string event_id,
    event_done_callback done_cb)
    : event_handle_cache_{event_handle_cache},
      handle_{event_handle_cache_.get_free_handle()},
      done_cb_{std::move(done_cb)},
      device_ptrs_{std::move(device_ptrs)},
      event_ids_{},
      stream_recorded_{stream} {
  if (!event_id.empty()) {
    event_ids_.emplace_back(std::move(event_id));
  }
}

void event::synchronize() const {
  PT_SYNHELPER_DEBUG("synchronizing event ", handle_);
  if (done_)
    PT_SYNHELPER_FATAL("Event ", this, " already done");
  auto status{synStatus::synSuccess};
  if (!is_partial())
    status = synEventSynchronize(handle_);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "Event synchronization failed");
  }
}

void event::complete() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (done_)
      PT_SYNHELPER_FATAL("Event ", this, " already done");
    done_ = true;
    if (done_cb_)
      done_cb_();
    if (handle_) {
      event_handle_cache_.release_handle(handle_);
      handle_ = nullptr;
    }
    done_cb_ = nullptr; // explicit destruction of cb to release any internally
                        // held objects
  }
  ready_var_.notify_all();
}

void event::stream_wait_event(stream& stream, const uint32_t flags) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (done_ || stream == stream_recorded_) {
    return;
  }

  auto status = synStreamWaitEvent(stream, handle_, flags);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "Recording of WaitEvent failed");
  }
}

void event::map_event_to_tensor(
    const synRecipeHandle recipe_handle,
    synLaunchTensorInfo* tensor_info) {
  auto status = synEventMapTensor(&handle_, 1, tensor_info, recipe_handle);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "synEventMapTensor failed");
  }
  is_partial_ = true;
}

event::~event() {
  if (!done_) {
    if (is_partial_) {
      PT_SYNHELPER_DEBUG(
          "Destroying partial event ", this, "that is not synchronized yet");
    } else {
      PT_SYNHELPER_FATAL(
          "Destroying event ", this, " that is not synchronized yet");
    }
  }
  if (handle_) {
    event_handle_cache_.release_handle(handle_);
  }
}
