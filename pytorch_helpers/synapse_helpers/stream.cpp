/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/stream.h"

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

#include "habana_helpers/logging.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/event.h"
#include "synapse_helpers/stream_event_manager.h"
#include "synapse_helpers/synapse_error.h"

using namespace synapse_helpers;

// stream flags are currently not supported
constexpr uint32_t STREAM_EMPTY_FLAGS = 0;

namespace {
synapse_error_v<synStreamType> convertInternalStreamType(stream_flavor flavor) {
  switch (flavor) {
    case DMA_D2D:
      return STREAM_TYPE_COPY_DEVICE_TO_DEVICE;
    case DMA_H2D:
      return STREAM_TYPE_COPY_HOST_TO_DEVICE;
    case DMA_D2H:
      return STREAM_TYPE_COPY_DEVICE_TO_HOST;
    case COMPUTE_0:
    case COMPUTE_1:
      // TODO: Need to add specifier for secondary stream of the same type
      return STREAM_TYPE_COMPUTE;
    case COLLECTIVE_0:
    case COLLECTIVE_1:
      // TODO: Need to add specifier for secondary stream of the same type
      return STREAM_TYPE_NETWORK_COLLECTIVE;
    case SEND:
      return STREAM_TYPE_NETWORK_SEND;
    case RECV:
      return STREAM_TYPE_NETWORK_RECEIVE;
    default:
      return synapse_error{
          "Unsupported stream flavor: " + std::to_string(flavor), synFail};
  }
}
} // namespace

namespace synapse_helpers {
template <typename collection_t>
void stream::try_sync_events(collection_t& events_to_sync) {
  for (auto& e : events_to_sync) {
    device_.synchronize_event(e);
  }
}

stream::stream(class device& device, stream_flavor flavor)
    : pending_cleanups_{},
      device_{device},
      mut_{},
      continue_{true},
      cond_var_{},
      handle_{nullptr} {
  gc_worker_ = std::thread(&stream::gc_thread_proc, this);
  auto syn_flavor = convertInternalStreamType(flavor);
  if (!ok(syn_flavor))
    PT_SYNHELPER_FATAL(
        "Stream type conversion failed with error: ",
        get_error(syn_flavor).error);
  auto status = synStreamCreate(
      &handle_, device_.id(), get_value(syn_flavor), STREAM_EMPTY_FLAGS);
  if (synStatus::synSuccess != status)
    PT_SYNHELPER_FATAL("Stream creation failed with status: ", status);
}

void stream::register_pending_event(const shared_event& event) {
  {
    std::lock_guard<std::mutex> lock_guard(mut_);
    auto status = synEventRecord(*event, handle_);
    if (synStatus::synSuccess != status) {
      PT_SYNHELPER_FATAL(
          "Event record failed on stream ", handle_, " with status: ", status);
    }
    pending_cleanups_.push_back(event);
  }
  cond_var_.notify_one();
}

void stream::gc_thread_proc() {
  while (true) {
    std::vector<shared_event> events_to_clean;
    {
      std::unique_lock<std::mutex> lock(mut_);
      gc_worker_is_busy_ = false;
      while (pending_cleanups_.empty() && continue_) {
        cond_var_.wait(lock);
      }
      if (!continue_) {
        break;
      }

      std::move(
          std::begin(pending_cleanups_),
          std::end(pending_cleanups_),
          std::back_inserter(events_to_clean));
      gc_worker_is_busy_ = true;
      pending_cleanups_.clear();
    }
    try_sync_events(events_to_clean);
  }
}

void stream::flush(int timeout_ms, int poll_rate_ms) {
  auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start <
         std::chrono::milliseconds(timeout_ms)) {
    if (!is_busy())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_rate_ms));
  }
}

stream::~stream() {
  std::unique_lock<std::mutex> lock(mut_);
  if (!continue_) {
    return;
  }
  continue_ = false;
  lock.unlock();
  cond_var_.notify_one();
  gc_worker_.join();
  lock.lock();
  try_sync_events(pending_cleanups_);
  pending_cleanups_.clear();
  synStreamSynchronize(handle_);
  synStreamDestroy(handle_);
}
} // namespace synapse_helpers
