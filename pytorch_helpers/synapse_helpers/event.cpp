/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/event.h"

#include <synapse_api.h>

#include <ostream>
#include <utility>

#include "habana_helpers/logging.h"
#include "synapse_helpers/event_handle_cache.h"
#include "synapse_helpers/stream.h"

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
    PT_SYNHELPER_FATAL("Event synchronization failed with: ", status);
  }
}

void event::complete() {
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
  ready_var_.notify_all();
}

void event::stream_wait_event(stream& stream, const uint32_t flags) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (done_ || stream == stream_recorded_) {
    return;
  }

  auto status = synStreamWaitEvent(stream, handle_, flags);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL("Recording of WaitEvent failed with: ", status);
  }
}

void event::map_event_to_tensor(
    const synRecipeHandle recipe_handle,
    synLaunchTensorInfo* tensor_info) {
  auto status = synEventMapTensorBase(&handle_, 1, tensor_info, recipe_handle);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_FATAL("synEventMapTensorBase failed with: ", status);
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
