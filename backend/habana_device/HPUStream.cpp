/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <iostream>
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/HPUStream.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"

namespace c10 {
namespace hpu {

static JoinEagerThreads join_eager_threads_cb = nullptr;
void setJoinEagerThreadsCB(JoinEagerThreads cb) {
  join_eager_threads_cb = cb;
}
void joinEagerThreadsCB() {
  if (join_eager_threads_cb != nullptr) {
    join_eager_threads_cb();
  }
}

namespace {

// Global stream state and constants
static std::once_flag init_flag;

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 57 bits --  -- 5 bits -----  -- 3 bits --
// zeros          stream id index  StreamIdType
//
// Where StreamIdType:
//  000 = default stream or externally allocated if id[63:3] != 0
//

// Thread-local current streams
static thread_local std::unique_ptr<StreamId> current_streams = nullptr;

// Populates global values.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  habana::HPURegistrar::get_device();
}

// Init front-end to ensure initialization only occurs once
static void initHPUStreamsOnce() {
  PT_DEVICE_DEBUG("STREAM:: HPUStream::initHPUStreamsOnce");
  // Inits default streams (once, globally)
  std::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams = std::make_unique<StreamId>();
  *current_streams = 0;
}

HPUStream HPUStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return HPUStream(
      HPUStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::HPU, device_index),
          stream_id));
}

} // anonymous namespace

HPUStream getStreamByStreamPtr(
    synapse_helpers::hpuStream_t ext_stream,
    DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return HPUStreamForId(device_index, (StreamId)ext_stream);
}

bool HPUStream::query() const {
  DeviceGuard guard{stream_.device()};
  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto hpu_stream_id = stream();
  auto device_index = device.id();
  PT_DEVICE_DEBUG(
      "STREAM:: Query User stream id ::",
      stream_.id(),
      " Hpu stream Index::",
      hpu_stream_id);
  auto& stream = device.get_stream(hpu_stream_id);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1) { /* only for lazy mode */
    /*TDB check if StepMarker is required for query */
    if (id() != getCurrentHPUStream(device_index).id()) {
      PT_IRGRAPH_DEBUG("step marker due to HPUStream::query");
      habana_lazy::HbLazyTensor::StepMarker({});
    } else {
      // If there are current jobs in stream. return false
      auto context =
          habana_lazy::get_device_lazy_execution_context(device_index);
      if (context->HaveJobsInStream(hpu_stream_id)) {
        return false;
      }
    }
  } else if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    joinEagerThreadsCB();
  }
  if (hpu_stream_id == 0) {
    return device.query_default_stream();
  } else {
    auto status = stream.query();
    if (status == synSuccess)
      return true;
    else
      PT_DEVICE_DEBUG(
          Logger::formatStatusMsg(status), "STREAM:: synStreamQuery");

    return false;
  }
}

void HPUStream::synchronize() const {
  DeviceGuard guard{stream_.device()};
  auto& device = habana::HPURegistrar::get_device().syn_device();
  auto hpu_stream_id = stream();
  auto device_index = device.id();
  PT_DEVICE_DEBUG(
      "STREAM:: synchronize User stream id ::",
      stream_.id(),
      " Hpu stream Index::",
      hpu_stream_id);
  auto& stream = device.get_stream(hpu_stream_id);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1) {
    if (id() != getCurrentHPUStream(device_index).id()) {
      PT_IRGRAPH_DEBUG("step marker due to HPUStream::synchronize");
      habana_lazy::HbLazyTensor::StepMarker({});
    } else {
      bool is_main_thread = habana::HPURegistrar::get_main_thread_id() ==
          std::this_thread::get_id();
      // If synchronize is called from userthread, just do wait till the
      // execution is over
      PT_IRGRAPH_DEBUG("step marker due to HPUStream::synchronize userthread");
      habana_lazy::HbLazyTensor::StepMarkerFinish(!is_main_thread);
    }
  } else if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    joinEagerThreadsCB();
  }
  if (hpu_stream_id == 0) {
    device.synchronize_default_stream();
  } else {
    stream.synchronize();
  }
}
// See Note [StreamId assignment]
synapse_helpers::hpuStream_t HPUStream::stream() const {
  return stream_.id();
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
HPUStream getStreamFromPool(
    const bool isHighPriority,
    DeviceIndex device_index) {
  initHPUStreamsOnce();
  if (device_index == -1) {
    auto& device = habana::HPURegistrar::get_device().syn_device();
    device_index = device.id();
  }

  // create stream
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device().syn_device();
  synapse_helpers::hpuStream_t stream;
  PT_DEVICE_DEBUG("STREAM:: create a new stream::");
  device.create_stream(stream, isHighPriority);

  PT_DEVICE_DEBUG(
      "STREAM:: HPUStream::getStreamFromPool got with stream index::", stream);
  return HPUStreamForId(device_index, stream);
}

HPUStream getDefaultHPUStream(DeviceIndex device_index) {
  initHPUStreamsOnce();
  if (device_index == -1) {
    auto& device = habana::HPURegistrar::get_device();
    device_index = device.id();
  }
  return HPUStreamForId(device_index, 0);
}

HPUStream getCurrentHPUStream(DeviceIndex device_index) {
  initHPUStreamsOnce();
  if (device_index == -1) {
    auto& device = habana::HPURegistrar::get_device();
    device_index = device.id();
  }
  PT_DEVICE_DEBUG(
      "STREAM:: getCurrentHPUStream current stream::", *current_streams);
  return HPUStreamForId(device_index, *current_streams);
}

void setCurrentHPUStream(HPUStream stream) {
  initHPUStreamsOnce();
  PT_DEVICE_DEBUG(
      "STREAM:: setCurrentHPUStream current stream::", *current_streams);
  if (*current_streams != stream.id()) {
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1) {
      habana_lazy::HbLazyTensor::StepMarkerBind();
    } else if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
      joinEagerThreadsCB();

      if (*current_streams == 0) {
        auto& device = habana::HPURegistrar::get_device().syn_device();
        synapse_helpers::hpuEvent_t id = device.create_event(false);
        device.record_event(id, *current_streams);
        device.wait_event(id, stream.id());
        device.delete_event(id, false);
      }
    }
    PT_DEVICE_DEBUG(
        "STREAM:: setCurrentHPUStream current stream::",
        *current_streams,
        " To stream::",
        stream.id());

    *current_streams = stream.id();
  }
}

std::ostream& operator<<(std::ostream& stream, const HPUStream& s) {
  return stream << s.unwrap();
}

} // namespace hpu
} // namespace c10
