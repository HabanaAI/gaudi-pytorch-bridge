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
#pragma once

#include <cstdint>
#include <utility>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
//#include "hpu_cached_devices.h"

#include <synapse_api_types.h>
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"

//#include "habana_lazy/hpu_lazy_tensors.h"

#include "pytorch_helpers/habana_helpers/pt_version_check.h"

/*
 * Stream pool note.
 *
 * A HPUStream is an abstraction of an actual synapse Stream on the HPU.
 * HPUStreams are backed by synapse Streams, but they use several pools
 * to minimize the costs associated with creating, retaining,
 * and destroying synapse Streams.
 *
 * There are three pools per device, and a device's pools are lazily created.
 *
 * The first pool contains only the default stream. When the default stream
 * is requested it's returned.
 *
 * The second pool is the "low priority" or "default priority" streams.
 * There are 32 of these streams per device, and when a stream is requested
 * one of these streams is returned round-robin. That is, the first stream
 * requested is at index 0, the second at index 1...to index 31,
 * then index 0 again.
 *
 * This means that if 33 low priority streams are requested, the first and
 * last streams requested are actually the same stream (under the covers)
 * and kernels enqueued on them cannot run concurrently.
 *
 * The third pool is the "high priority" streams. The third pool acts like
 * the second pool except the streams are created with a higher priority.
 *
 * These pools suggest that stream users should prefer many short-lived streams,
 * as the cost of acquiring and releasing streams is effectively zero. If
 * many longer-lived streams are required in performance critical scenarios
 * then the functionality here may need to be extended to allow, for example,
 * "reserving" a subset of the pool so that other streams do not accidentally
 * overlap the performance critical streams.
 *
 * Note: although the notion of "current stream for device" is thread local
 * (every OS thread has a separate current stream, as one might expect),
 * the stream pool is global across all threads; stream 0 is always stream 0
 * no matter which thread you use it on.  Multiple threads can synchronize
 * on the same stream.  Although the HPU documentation is not very clear
 * on the matter, streams are thread safe; e.g., it is safe to enqueue
 * a kernel on the same stream from two different threads.
 *
 * Currently on HPU priority is not supported. all the streams have equal
 * priority
 */

typedef void (*JoinEagerThreads)(void);

namespace c10 {
namespace hpu {

// Value object representing a HPU stream.  This is just a wrapper
// around c10::Stream, but it comes with a little extra HPU-specific
// functionality (conversion to device stream), and a guarantee that
// the wrapped c10::Stream really is a HPU stream.
class HPUStream {
 public:
  enum Unchecked { UNCHECKED };

  /// Construct a HPUStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a HPU stream.
  explicit HPUStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::HPU);
  }

  /// Construct a HPUStream from a Stream with no error checking.
  /// This constructor uses the "named" constructor idiom, and can
  /// be invoked as: HPUStream(HPUStream::UNCHECKED, stream)
  explicit HPUStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const HPUStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const HPUStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to hpuStream_t.
  operator synapse_helpers::hpuStream_t() const {
    return stream();
  }

  /// Implicit conversion to Stream (a.k.a., forget that the stream is a
  /// HPU stream).
  operator Stream() const {
    return unwrap();
  }

  /// Used to avoid baking in device type explicitly to Python-side API.
  DeviceType device_type() const {
    return DeviceType::HPU;
  }

  /// Get the HPU device index that this stream is associated with.
  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a HPU device.
  Device device() const {
    return Device(DeviceType::HPU, device_index());
  }

  /// Return the stream ID corresponding to this particular stream.
  StreamId id() const {
    return stream_.id();
  }

  bool query() const;

  void synchronize() const;

  int priority() const {
    DeviceGuard guard{stream_.device()};
    int priority = 0;
    // TDB not supported now.
    // syncStreamGetPriority(stream(), &priority));
    return priority;
  }

  /// Explicit conversion to hpuStream_t.
  synapse_helpers::hpuStream_t stream() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return stream_;
  }

  /// Reversibly pack a HPUStream into a uint64_t representation.  This may
  /// be helpful when storing a HPUStream in a C struct, where you cannot
  /// conveniently place the HPUStream object itself (which is morally
  /// equivalent, but unfortunately is not POD due to the fact that it
  /// has constructors.)
  ///
  /// The HPUStream can be unpacked using unpack().  The format of
  /// the uint64_t is unspecified and may be changed.
  struct c10::StreamData3 pack() const {
    return stream_.pack3();
  }

  // Unpack a HPUStream from the 3 fields generated by pack().
  static HPUStream unpack3(
      StreamId stream_id,
      DeviceIndex device_index,
      DeviceType device_type) {
    return HPUStream(Stream::unpack3(stream_id, device_index, device_type));
  }

  // TBD not supported
  static std::tuple<int, int> priority_range() {
    /*
    // Note: this returns the range of priority **supported by PyTorch**, not
    // the range of priority **supported by HPU**. The former is a subset of
    // the latter. Currently PyTorch only supports 0 and -1, which are "low" and
    // "high" priority.
    int least_priority, greatest_priority;
    C10_HPU_CHECK(
        hpuDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    TORCH_INTERNAL_ASSERT(
        least_priority >= 0, "Unexpected HPU stream priority range");
    TORCH_INTERNAL_ASSERT(
        greatest_priority <= -1, "Unexpected HPU stream priority range");
    */
    return std::make_tuple(0, -1);
  }

  // Deleted for now; use HPUEvent::block instead
  // void synchronize_with(const HPUEvent& event) const;
 private:
  Stream stream_;
};

void setJoinEagerThreadsCB(JoinEagerThreads cb);
void joinEagerThreadsCB();

/**
 * Get a new stream from the HPU stream pool.  You can think of this
 * as "creating" a new stream, but no such creation actually happens;
 * instead, streams are preallocated from the pool and returned in a
 * round-robin fashion.
 *
 * You can request a stream from the high priority pool by setting
 * isHighPriority to true, or a stream for a specific device by setting device
 * (defaulting to the current HPU stream.)
 */
TORCH_API HPUStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

/**
 * Get the default HPU stream, for the passed HPU device, or for the
 * current device if no device index is passed.  The default stream is
 * where most computation occurs when you aren't explicitly using
 * streams.
 */
TORCH_API HPUStream getDefaultHPUStream(DeviceIndex device_index = -1);

/**
 * Get the current HPU stream, for the passed HPU device, or for the
 * current device if no device index is passed.  The current HPU stream
 * will usually be the default HPU stream for the device, but it may
 * be different if someone called 'setCurrentHPUStream' or used 'StreamGuard'
 * or 'HPUStreamGuard'.
 */
TORCH_API HPUStream getCurrentHPUStream(DeviceIndex device_index = -1);

/**
 * Set the current stream on the device of the passed in stream to be
 * the passed in stream.  Yes, you read that right: this function
 * has *nothing* to do with the current device: it toggles the current
 * stream of the device of the passed stream.
 *
 * Confused?  Avoid using this function; prefer using 'HPUStreamGuard' instead
 * (which will switch both your current device and current stream in the way you
 * expect, and reset it back to its original state afterwards).
 */
TORCH_API void setCurrentHPUStream(HPUStream stream);

C10_API std::ostream& operator<<(std::ostream& stream, const HPUStream& s);

/**
 * Get a HPUStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
TORCH_API HPUStream getStreamByStreamPtr(
    synapse_helpers::hpuStream_t ext_stream,
    DeviceIndex device_index);

} // namespace hpu
} // namespace c10

namespace std {
template <>
struct hash<c10::hpu::HPUStream> {
  size_t operator()(c10::hpu::HPUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
