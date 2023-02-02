/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <iostream>
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/habana_device/HPUStream.h"

namespace c10 {
namespace hpu {

namespace {

// Global stream state and constants
static std::once_flag init_flag;

// Non-default streams
static std::once_flag device_flags;

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
  synapse_helpers::HPURegistrar::get_device();
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

bool HPUStream::query() const {
  DeviceGuard guard{stream_.device()};
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto hpu_stream_id = stream();
  auto device_index = device.id();
  PT_DEVICE_DEBUG(
      "STREAM:: Query User stream id ::",
      stream_.id(),
      " Hpu stream Index::",
      hpu_stream_id);
  auto& stream = device.get_stream(hpu_stream_id);
  /*TDB check if StepMarker is required for query */
  if (id() != getCurrentHPUStream(device_index).id()) {
    habana_lazy::HbLazyTensor::StepMarker({});
  } else {
    bool is_main_thread = synapse_helpers::HPURegistrar::getMainThreadId() ==
        std::this_thread::get_id();
    // If query is called from userthread, just do wait till the execution
    // is over
    habana_lazy::HbLazyTensor::StepMarkerFinish(!is_main_thread);
  }
  auto status = stream.query();
  if (status == synSuccess)
    return true;
  else
    PT_DEVICE_DEBUG("STREAM:: synStreamQuery failed with status", status);

  return false;
}

void HPUStream::synchronize() const {
  DeviceGuard guard{stream_.device()};
  auto& device = synapse_helpers::HPURegistrar::get_device();
  auto hpu_stream_id = stream();
  auto device_index = device.id();
  PT_DEVICE_DEBUG(
      "STREAM:: synchronize User stream id ::",
      stream_.id(),
      " Hpu stream Index::",
      hpu_stream_id);
  auto& stream = device.get_stream(hpu_stream_id);
  if (id() != getCurrentHPUStream(device_index).id()) {
    habana_lazy::HbLazyTensor::StepMarker({});
  } else {
    bool is_main_thread = synapse_helpers::HPURegistrar::getMainThreadId() ==
        std::this_thread::get_id();
    // If synchronize is called from userthread, just do wait till the execution
    // is over
    habana_lazy::HbLazyTensor::StepMarkerFinish(!is_main_thread);
  }
  stream.synchronize();
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
    auto& device = synapse_helpers::HPURegistrar::get_device();
    device_index = device.id();
  }

  // create stream
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
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
    auto& device = synapse_helpers::HPURegistrar::get_device();
    device_index = device.id();
  }
  return HPUStreamForId(device_index, 0);
}

HPUStream getCurrentHPUStream(DeviceIndex device_index) {
  initHPUStreamsOnce();
  if (device_index == -1) {
    auto& device = synapse_helpers::HPURegistrar::get_device();
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
    habana_lazy::HbLazyTensor::StepMarkerBind();
    *current_streams = stream.id();
  }
}

std::ostream& operator<<(std::ostream& stream, const HPUStream& s) {
  return stream << s.unwrap();
}

} // namespace hpu
} // namespace c10
