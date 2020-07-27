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
      // TODO: Need to add specifier for secondary stream of the same type
      return STREAM_TYPE_COMPUTE;
    case COLLECTIVE_0:
      // TODO: Need to add specifier for secondary stream of the same type
      return STREAM_TYPE_NETWORK_COLLECTIVE;
    default:
      return synapse_error{
          "Unsupported stream flavor: " + std::to_string(flavor), synFail};
  }
}
} // namespace

namespace synapse_helpers {
stream::stream(class device& device, stream_flavor flavor)
    : pending_cleanups_{},
      device_{device},
      mut_{},
      cond_var_{},
      handle_{nullptr} {
  pending_cleanups_.push({});
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
    pending_cleanups_.push(event);
  }
  cond_var_.notify_one();
}

void stream::gc_thread_proc() {
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
    device_.synchronize_event(event_to_clean);
  }
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
  synStreamDestroy(handle_);
}
} // namespace synapse_helpers
