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
#include "backend/synapse_helpers/event.h"

#include <synapse_api.h>

#include <ostream>
#include <utility>

#include "backend/synapse_helpers/event_handle_cache.h"
#include "backend/synapse_helpers/stream.h"
#include "habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/python_utils.h"

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
  habana_helpers::AutoNoGIL gil_release;
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
  habana_helpers::AutoNoGIL gil_release;
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

void event::wait() {
  habana_helpers::AutoNoGIL gil_release;
  std::unique_lock<std::mutex> lock(mutex_);
  ready_var_.wait(lock, [this]() -> bool { return done(); });
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
