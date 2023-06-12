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

#include <c10/core/DeviceGuard.h>
#include <c10/util/Exception.h>

#include <synapse_api_types.h>
#include "HPUStream.h"
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"

#include <cstdint>
#include <utility>

namespace at {
namespace hpu {

/*
 * HPUEvents are movable not copyable wrappers around HPU's events.
 *
 * HPUEvents are constructed lazily when first recorded. The event has
 * a device, and this device is acquired from the first recording stream.
 * However, if reconstructed from a id, the device should be explicitly
 * specified; it will use the current device. Later streams that record the
 * event must match this device.
 */
struct HPUEvent {
  // Constructors
  // Default value for `flags` is specified below - it's 0 disable collect time
  HPUEvent() {}
  HPUEvent(unsigned int flags) : flags_{flags} {}

  // Note: event destruction done on creating device to avoid creating a
  // HPU context on other devices.
  ~HPUEvent() {
    auto& dev = synapse_helpers::HPURegistrar::get_device();
    if (is_created_) {
      dev.delete_event(id_, flags_);
    }
  }

  HPUEvent(const HPUEvent&) = delete;
  HPUEvent& operator=(const HPUEvent&) = delete;

  HPUEvent(HPUEvent&& other) {
    moveHelper(std::move(other));
  }
  HPUEvent& operator=(HPUEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  operator synapse_helpers::hpuEvent_t() const {
    return id();
  }

  // Less than operator (to allow use in sets)
  friend bool operator<(const HPUEvent& left, const HPUEvent& right) {
    return left.id_ < right.id_;
  }

  bool isCreated() const {
    return is_created_;
  }

  Device device() const {
    return Device(DeviceType::HPU, device_index_);
  }

  DeviceIndex device_index() const {
    return device_index_;
  }

  synapse_helpers::hpuEvent_t id() const {
    return id_;
  }

  // Note: hpuEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }
    auto& device = synapse_helpers::HPURegistrar::get_device();

    return device.query_event(id_);
  }

  void record() {
    record(c10::hpu::getCurrentHPUStream());
  }

  void recordOnce(const c10::hpu::HPUStream& stream) {
    if (!was_recorded_)
      record(stream);
  }

  // Note: hpuEventRecord must be called on the same device as the event.
  void record(const c10::hpu::HPUStream& stream) {
    auto& device = synapse_helpers::HPURegistrar::get_device();
    if (!is_created_) {
      createEvent(stream.device_index());
      created_with_stream_ = stream.stream();
      PT_DEVICE_DEBUG("event id::", id_);
    }

    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");

    // if the current stream and the record stream is different
    // just add a event record without a step_marker.
    // if the current stream is same as record stream, then
    // do mark step and use this for launch.
    PT_DEVICE_DEBUG(
        "Event Recod Current_stream::",
        (c10::hpu::getCurrentHPUStream()).stream(),
        " record stream::",
        stream.stream());
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1) {
      if (stream.stream() == (c10::hpu::getCurrentHPUStream()).stream()) {
        PT_DEVICE_DEBUG("Reocrd Stream current and record stream are same");
        habana_lazy::HbLazyTensor::StepMarker({});
      }
    }

    device.record_event(id_, stream.stream());
    recorded_stream_ = stream.stream();
    was_recorded_ = true;
  }

  // Note: hpuStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual HPU resources associated with it.
  void block(const c10::hpu::HPUStream& stream) {
    if (is_created_) {
      if (stream.stream() == recorded_stream_) {
        return;
      }
      auto& device = synapse_helpers::HPURegistrar::get_device();
      device.wait_event(id_, stream.stream());
    }
  }

  // Note: hpuEventElapsedTime can be safely called from any device
  float elapsed_time(const HPUEvent& other) const {
    TORCH_CHECK(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    auto& device = synapse_helpers::HPURegistrar::get_device();
    return device.eplased_time(id_, other.id_);
  }

  // Note: hpuEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      auto& device = synapse_helpers::HPURegistrar::get_device();
      device.synchronize_event(id_);
    }
  }

 private:
  // flags_ is used to create the event to enable/disable capture timing
  // it can take value 0(disable capture time) or EVENT_COLLECT_TIME=1
  //(enable capture time)
  unsigned int flags_ = 0;
  bool is_created_ = false;
  bool was_recorded_ = false;
  DeviceIndex device_index_ = -1;
  synapse_helpers::hpuEvent_t id_ = 0;
  synapse_helpers::hpuStream_t recorded_stream_;
  synapse_helpers::hpuStream_t created_with_stream_;

  void createEvent([[maybe_unused]] DeviceIndex device_index) {
    // get device
    auto& dev = synapse_helpers::HPURegistrar::get_device();
    device_index_ = dev.id();
    id_ = dev.get_event_index();
    dev.create_event(id_, flags_);
    is_created_ = true;
    PT_DEVICE_DEBUG("created event with ::", id_);
  }

  void moveHelper(HPUEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(id_, other.id_);
    std::swap(recorded_stream_, other.recorded_stream_);
  }
}; // namespace hpu

} // namespace hpu
} // namespace at
