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
#include "backend/synapse_helpers/stream.h"

#include <synapse_api.h>
#include <synapse_common_types.h>

#include <absl/types/variant.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/event.h"
#include "backend/synapse_helpers/stream_event_manager.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "habana_helpers/logging.h"

using namespace synapse_helpers;

// stream flags are currently not supported
constexpr uint32_t STREAM_EMPTY_FLAGS = 0;

namespace synapse_helpers {
stream::stream(class device& device)
    : pending_cleanups_{},
      device_{device},
      mut_{},
      cond_var_{},
      handle_{nullptr} {
  pending_cleanups_.push({});
  gc_worker_ = std::thread(&stream::gc_thread_proc, this);
  auto status =
      synStreamCreateGeneric(&handle_, device_.id(), STREAM_EMPTY_FLAGS);
  if (synStatus::synSuccess != status)
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "Stream creation failed.");
  PT_SYNHELPER_DEBUG("Stream creation with handle: ", handle_);
}

void stream::register_pending_event(
    const shared_event& event,
    bool record_event) {
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    if (record_event) {
      auto status = synEventRecord(*event, handle_);
      if (synStatus::synSuccess != status) {
        PT_SYNHELPER_FATAL(
            Logger::formatStatusMsg(status),
            "Event record failed on stream ",
            handle_);
      }
    }
    pending_cleanups_.push(event);
  }
  cond_var_.notify_one();
}

void stream::gc_thread_proc() {
  PT_SYNHELPER_DEBUG("GC thread stream::");
  std::queue<shared_event> partial_events{};
  while (true) {
    std::unique_lock<std::mutex> lock(mut_);

    pending_cleanups_.pop();
    if (pending_cleanups_.empty()) {
      cond_var_empty.notify_all();
    }
    cond_var_.wait(lock, [this] { return !pending_cleanups_.empty(); });

    auto event_to_clean = pending_cleanups_.front();
    if (!event_to_clean) {
      break;
    }
    lock.unlock();
    // Delay synchronizing partial events until the next real event. a real can
    // only be enqueued after recipe execution. Once it is triggered we know all
    // preceeding partial event must be triggered.
    if (event_to_clean->is_partial()) {
      partial_events.push(std::move(event_to_clean));
    } else {
      device_.synchronize_event(event_to_clean);
      // Call synchronize_event on all preceeding partial events to trigger
      // event done callback
      while (!partial_events.empty()) {
        device_.synchronize_event(partial_events.front());
        partial_events.pop();
      }
    }
  }
}

void stream::synchronize() {
  synStatus status = synStreamSynchronize(handle_);
  if (synStatus::synSuccess != status)
    PT_SYNHELPER_FATAL(
        Logger::formatStatusMsg(status), "synStreamSynchronize failed.");
}

synStatus stream::query() {
  synStatus status = synStreamQuery(handle_);
  if (synStatus::synSuccess != status)
    PT_SYNHELPER_DEBUG(
        Logger::formatStatusMsg(status), "synStreamSynchronize failed.");
  return status;
}

void stream::flush(int timeout_ms) {
  std::unique_lock<std::mutex> lock(mut_);
  cond_var_empty.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
    return pending_cleanups_.empty();
  });
}

stream::~stream() {
  std::unique_lock<std::mutex> lock(mut_);
  pending_cleanups_.push({});
  lock.unlock();
  cond_var_.notify_one();
  gc_worker_.join();
  synStreamSynchronize(handle_);
  PT_SYNHELPER_DEBUG("Stream dtor handle: ", handle_);
  synStreamDestroy(handle_);
}
} // namespace synapse_helpers
