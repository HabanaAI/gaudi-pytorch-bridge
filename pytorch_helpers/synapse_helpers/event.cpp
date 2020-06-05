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

#include "synapse_helpers/event_handle_cache.h"
#include "synapse_helpers/logging.h"
#include "synapse_helpers/stream.h"

using namespace synapse_helpers;

event::event(event_handle_cache& event_handle_cache, stream& stream, event_done_callback done_cb)
    : event_handle_cache_{event_handle_cache},
      handle_{event_handle_cache_.get_free_handle()},
      done_cb_{std::move(done_cb)},
      stream_recorded_{stream} {}

synStatus event::synchronize() {
  std::unique_lock<std::mutex> sync_lock(sync_mutex_);
  std::unique_lock<std::mutex> lock(mutex_);
  if (done_) return synSuccess;
  lock.unlock();
  auto status = synEventSynchronize(handle_);
  lock.lock();
  if (synStatus::synSuccess != status) {
    LOG_(FATAL) << "Event synchronization failed with: " << status;
  }
  done_ = true;
  if (done_cb_) done_cb_();
  if (handle_) {
    event_handle_cache_.release_handle(handle_);
    handle_ = nullptr;
  }
  done_cb_ = nullptr;  // explicit destruction of cb to release any internally held objects
  return status;
}

bool event::streamWaitEvent(stream& stream, const uint32_t flags) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (!done_ && stream != stream_recorded_) {
    auto status = synStreamWaitEvent(stream, handle_, flags);
    if (synStatus::synSuccess != status) {
      LOG_(FATAL) << "Recording of WaitEvent failed with: " << status;
    }
    return true;
  }
  return false;
}

event::~event() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!done_) {
    lock.unlock();
    auto status = synchronize();
    if (synStatus::synSuccess != status) {
      LOG_(FATAL) << "Event synchronization failed at destruction of an event.";
    }
    lock.lock();
  }
  if (handle_) {
    event_handle_cache_.release_handle(handle_);
  }
}
