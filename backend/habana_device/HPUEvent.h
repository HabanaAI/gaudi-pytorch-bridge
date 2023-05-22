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
 * However, if reconstructed from a handle, the device should be explicitly
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
    auto& dev = habana::HPURegistrar::get_device().syn_device();
    if (is_created_) {
      if (created_with_stream_ == 0 &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
        dev.delete_event_default_stream(handle_, flags_);
      } else {
        if (flags_) {
          dev.get_time_event_handle_cache().release_handle(handle_);
        } else {
          dev.get_event_handle_cache().release_handle(handle_);
        }
      }
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

  operator synEventHandle() const {
    return handle();
  }

  // Less than operator (to allow use in sets)
  friend bool operator<(const HPUEvent& left, const HPUEvent& right) {
    return left.handle_ < right.handle_;
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
  synEventHandle handle() const {
    return handle_;
  }

  // Note: hpuEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }
    auto& device = habana::HPURegistrar::get_device().syn_device();

    if (created_with_stream_ == 0 &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
      return device.query_event_default_stream(handle_);
    } else {
      auto status = synEventQuery(handle_);
      if (status == synSuccess) {
        return true;
      } else {
        PT_DEVICE_DEBUG(
            Logger::formatStatusMsg(status), "STREAM:: synEventQuery");
      }
    }

    return false;
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
    auto& device = habana::HPURegistrar::get_device().syn_device();
    if (!is_created_) {
      createEvent(stream.device_index());
      created_with_stream_ = stream.stream();
      if (created_with_stream_ == 0 &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
        device.create_default_stream_event(handle_, flags_);
      }
      PT_DEVICE_DEBUG("reusing event handle::", handle_);
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
    if (created_with_stream_ == 0 &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
      device.record_event_default_stream(handle_);
    } else {
      auto status = synEventRecord(handle_, device.get_stream(stream.stream()));
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL(
            Logger::formatStatusMsg(status), "synEventRecord failed");
      }
    }
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
      auto& device = habana::HPURegistrar::get_device().syn_device();
      if (stream.stream() == 0 &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
        device.wait_event_default_stream(handle_);
      } else {
        auto status =
            synStreamWaitEvent(device.get_stream(stream.stream()), handle_, 0);
        if (synStatus::synSuccess != status) {
          PT_DEVICE_FATAL(
              Logger::formatStatusMsg(status), "synStreamWaitEvent failed");
        }
      }
    }
  }

  // Note: hpuEventElapsedTime can be safely called from any device
  float elapsed_time(const HPUEvent& other) const {
    TORCH_CHECK(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    auto& device = habana::HPURegistrar::get_device().syn_device();
    uint64_t time_ms = 0;
    if (created_with_stream_ == 0 &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
      time_ms = device.eplased_time_default_stream(handle_, other.handle_);
    } else {
      auto status = synEventElapsedTime(&time_ms, handle_, other.handle_);
      if (synStatus::synSuccess != status) {
        PT_DEVICE_DEBUG(
            Logger::formatStatusMsg(status), "synEventElapsedTime failed");
      }
    }
    return time_ms;
  }

  // Note: hpuEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      auto& device = habana::HPURegistrar::get_device().syn_device();
      if (created_with_stream_ == 0 &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
        device.synchronize_event_default_stream(handle_);
      } else {
        auto status = synEventSynchronize(handle_);
        if (synStatus::synSuccess != status) {
          PT_DEVICE_FATAL(
              Logger::formatStatusMsg(status), "synEventSynchronize failed");
        }
      }
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
  synEventHandle handle_ = {nullptr};
  synapse_helpers::hpuStream_t recorded_stream_;
  synapse_helpers::hpuStream_t created_with_stream_;

  void createEvent([[maybe_unused]] DeviceIndex device_index) {
    // get device
    auto& dev = habana::HPURegistrar::get_device().syn_device();
    device_index_ = dev.id();
    if (flags_) {
      handle_ = dev.get_time_event_handle_cache().get_free_handle();
    } else {
      handle_ = dev.get_event_handle_cache().get_free_handle();
    }
    is_created_ = true;
  }

  void moveHelper(HPUEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(handle_, other.handle_);
    std::swap(recorded_stream_, other.recorded_stream_);
  }
}; // namespace hpu

} // namespace hpu
} // namespace at
