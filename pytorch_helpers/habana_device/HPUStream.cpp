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
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
// static constexpr unsigned int kDefaultFlags = hpuStreamNonBlocking;
static constexpr int kStreamTypeBits = 3;

// Note: lower numbers are higher priorities, zero is default priority
// static constexpr int kHighPriority = -1;
static constexpr int kLowPriority = 0;

// Non-default streams
// Note: the number of HPU devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in HPUStream.h).
// The streams are "leaked": they are created but never destroyed because the
// destruction of global variables could happen after the HPU runtime has
// already been destroyed and thus invoking hpuStreamDestroy could lead to a
// crash. It's likely an issue in HPU, but to be safe - let's just "forget"
// the destruction.
static std::once_flag device_flags;
static std::atomic<uint32_t> low_priority_counters;
// TDB static std::atomic<uint32_t> high_priority_counters;
static synapse_helpers::hpuStream_t low_priority_streams[kStreamsPerPool] = {0};
// TDB static hpuStream_t high_priority_streams[MAX_DEVICES_PER_BOX]
//                                         [kStreamsPerPool];

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 57 bits --  -- 5 bits -----  -- 3 bits --
// zeros          stream id index  StreamIdType
//
// Where StreamIdType:
//  000 = default stream or externally allocated if id[63:3] != 0
//  001 = low priority stream
//  010 = high priority stream
//
// This is not really for efficiency; it's just easier to write the code
// to extract the index if we do this with bitmasks :)
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream.  However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.
//
enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  LOW = 0x1,
  // HIGH = 0x2,
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT:
      stream << "DEFAULT";
      break;
    case StreamIdType::LOW:
      stream << "LOW";
      break;
    /*case StreamIdType::HIGH:
      stream << "HIGH";
      break;*/
    default:
      stream << static_cast<uint8_t>(s);
      break;
  }
  return stream;
}

// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]

static inline StreamIdType streamIdType(StreamId s) {
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  return static_cast<StreamIdType>(s & mask_for_type);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> kStreamTypeBits) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(si) << kStreamTypeBits) |
      static_cast<StreamId>(st);
}

// Thread-local current streams
static thread_local std::unique_ptr<StreamId> current_streams = nullptr;

// Populates global values.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  synapse_helpers::HPURegistrar::get_device();
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(UNUSED DeviceIndex device_index) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  int num_compute_stream = device.get_compute_stream_count() - 1;
  PT_DEVICE_DEBUG(
      "STREAM:: HPUStream::Init Compute stream::.", num_compute_stream);

  for (const auto i : c10::irange(num_compute_stream)) {
    synapse_helpers::hpuStream_t lowpri_stream;
    PT_DEVICE_DEBUG("STREAM:: create a new compute stream::", i);

    device.create_compute_stream(lowpri_stream);
    low_priority_streams[i] = lowpri_stream;
    // device.create_compute_stream(hipri_stream);
  }

  low_priority_counters = 0;
  // high_priority_counters = 0;
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
  *current_streams = makeStreamId(StreamIdType::DEFAULT, 0);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in HPUStream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  int num_compute_stream = device.get_compute_stream_count() - 1;
  auto raw_idx = counter++;
  PT_DEVICE_DEBUG(
      "STREAM:: HPUStream::streamm count::",
      num_compute_stream,
      "stream index",
      raw_idx);
  return raw_idx % num_compute_stream;
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
  auto si = streamIdIndex(stream_.id());
  auto& stream = device.get_compute_stream(si);
  PT_DEVICE_DEBUG("STREAM:: Query stream id ::", stream_.id());
  /*TDB check if StepMarker is required for query */
  habana_lazy::HbLazyTensor::StepMarker({});
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
  auto si = streamIdIndex(stream_.id());
  auto& stream = device.get_compute_stream(si);
  habana_lazy::HbLazyTensor::StepMarker({});
  stream.synchronize();
}
// See Note [StreamId assignment]
synapse_helpers::hpuStream_t HPUStream::stream() const {
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  PT_DEVICE_DEBUG(
      "STREAM:: stream info stream type::", st, " stream index::", si);
  switch (st) {
    case StreamIdType::DEFAULT:
      TORCH_INTERNAL_ASSERT(
          si == 0,
          "Unrecognized stream ",
          stream_,
          " (I think this should be the default stream, but I got a non-zero index ",
          si,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that; use the",
          " official API like c10::hpu::getStreamFromPool() to get a new stream.");
      return 0;
    case StreamIdType::LOW:
      PT_DEVICE_DEBUG(
          "STREAM:: device hpu stream index::", low_priority_streams[si]);
      return low_priority_streams[si];
    /*case StreamIdType::HIGH:
      return high_priority_streams[si];*/
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
HPUStream getStreamFromPool(
    UNUSED const bool isHighPriority,
    DeviceIndex device_index) {
  initHPUStreamsOnce();
  if (device_index == -1) {
    auto& device = synapse_helpers::HPURegistrar::get_device();
    device_index = device.id();
  }

  // Initializes the stream pools (once)
  std::call_once(device_flags, initDeviceStreamState, device_index);

  const auto idx = get_idx(low_priority_counters);
  PT_DEVICE_DEBUG(
      "STREAM:: HPUStream::getStreamFromPool got with stream index::", idx);
  return HPUStreamForId(device_index, makeStreamId(StreamIdType::LOW, idx));
}

HPUStream getDefaultHPUStream(DeviceIndex device_index) {
  initHPUStreamsOnce();
  if (device_index == -1) {
    auto& device = synapse_helpers::HPURegistrar::get_device();
    device_index = device.id();
  }
  return HPUStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
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
