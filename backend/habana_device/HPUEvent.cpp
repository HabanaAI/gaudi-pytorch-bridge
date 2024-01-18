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

//#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/HPUEvent.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_eager/eager_context.h"
#include "habana_lazy/hpu_lazy_tensors.h"

namespace at {
namespace hpu {
HPUEvent::~HPUEvent() {
  auto& dev = habana::HPURegistrar::get_device().syn_device();
  if (is_created_) {
    dev.delete_event(id_, flags_);
  }
}

void HPUEvent::createEvent([[maybe_unused]] DeviceIndex device_index) {
  // get device
  auto& dev = habana::HPURegistrar::get_device().syn_device();
  device_index_ = dev.id();
  id_ = dev.create_event(flags_);
  is_created_ = true;
  PT_DEVICE_DEBUG("created event with ::", id_);
}

void HPUEvent::moveHelper(HPUEvent&& other) {
  std::swap(flags_, other.flags_);
  std::swap(is_created_, other.is_created_);
  std::swap(was_recorded_, other.was_recorded_);
  std::swap(device_index_, other.device_index_);
  std::swap(id_, other.id_);
  std::swap(recorded_stream_, other.recorded_stream_);
}

// Note: hpuEventQuery can be safely called from any device
bool HPUEvent::query() const {
  if (!is_created_) {
    return true;
  }
  auto& device = habana::HPURegistrar::get_device().syn_device();

  return device.query_event(id_);
}

void HPUEvent::record() {
  record(c10::hpu::getCurrentHPUStream());
}

void HPUEvent::recordOnce(const c10::hpu::HPUStream& stream) {
  if (!was_recorded_)
    record(stream);
}

// Note: hpuEventRecord must be called on the same device as the event.
void HPUEvent::record(const c10::hpu::HPUStream& stream) {
  auto& device = habana::HPURegistrar::get_device().syn_device();
  if (!is_created_) {
    createEvent(stream.device_index());
    created_with_stream_ = stream.stream();
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
      PT_IRGRAPH_DEBUG("step marker due to HPUEvent::record");
      habana_lazy::HbLazyTensor::StepMarker(
          {}, nullptr, {}, (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SFG) == true));
    }
  } else if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    c10::hpu::joinEagerThreadsCB();
  }

  device.record_event(id_, stream.stream());
  recorded_stream_ = stream.stream();
  was_recorded_ = true;
}

// Note: hpuStreamWaitEvent must be called on the same device as the stream.
// The event has no actual HPU resources associated with it.
void HPUEvent::block(const c10::hpu::HPUStream& stream) {
  if (is_created_) {
    if (stream.stream() == recorded_stream_) {
      return;
    }
    auto& device = habana::HPURegistrar::get_device().syn_device();
    device.wait_event(id_, stream.stream());
  }
}

// Note: hpuEventElapsedTime can be safely called from any device
float HPUEvent::elapsed_time(const HPUEvent& other) const {
  TORCH_CHECK(
      is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
  auto& device = habana::HPURegistrar::get_device().syn_device();
  return device.elapsed_time(id_, other.id_);
}

// Note: hpuEventSynchronize can be safely called from any device
void HPUEvent::synchronize() const {
  if (is_created_) {
    auto& device = habana::HPURegistrar::get_device().syn_device();
    device.synchronize_event(id_);
  }
}

} // namespace hpu
} // namespace at
