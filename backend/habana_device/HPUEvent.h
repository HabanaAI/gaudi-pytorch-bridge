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
 public:
  // Constructors
  // Default value for `flags` is specified below - it's 0 disable collect time
  explicit HPUEvent() {}
  explicit HPUEvent(unsigned int flags) : flags_{flags} {}

  ~HPUEvent();

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

  synapse_helpers::hpuEvent_t event() const {
    return id();
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
  bool query() const;

  void record();

  void recordOnce(const c10::hpu::HPUStream& stream);

  // Note: hpuEventRecord must be called on the same device as the event.
  void record(const c10::hpu::HPUStream& stream);

  void block(const c10::hpu::HPUStream& stream);

  // Note: hpuEventElapsedTime can be safely called from any device
  float elapsed_time(const HPUEvent& other) const;

  // Note: hpuEventSynchronize can be safely called from any device
  void synchronize() const;

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

  void createEvent([[maybe_unused]] DeviceIndex device_index);

  void moveHelper(HPUEvent&& other);
}; // namespace hpu

} // namespace hpu
} // namespace at
