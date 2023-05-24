/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "backend/habana_device/HPUDevice.h"
#include <memory>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/scalar_cache.h"
#include "backend/synapse_helpers/time_slot.h"

namespace habana {

HPUDevice::HPUDevice()
    : scalar_cache_{std::make_unique<backend::ScalarCache>()} {
  auto allocatorVar =
      [](synDeviceId id) -> std::unique_ptr<synapse_helpers::device_allocator> {
    return std::make_unique<habana::HPUAllocator>(id);
  };

  auto device_ptr_or_error = synapse_helpers::device::get_or_create(
      synapse_helpers::device::get_supported_devices(), allocatorVar);

  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          device_ptr_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(device_ptr_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  } else {
    auto device_ptr = absl::get<std::shared_ptr<synapse_helpers::device>>(
        device_ptr_or_error);
    device_ = std::move(device_ptr);
  }
}

HPUDevice::~HPUDevice() {
  // Theoretically another refernce can be kept elsewhere.
  if (device_.use_count() != 1) {
    TORCH_WARN(
        "when deleting HPUDevice, device is kept alive by another references ",
        device_.use_count());
  }
}

std::shared_ptr<synapse_helpers::TimeSlot> HPUDevice::create_time_slot(
    synapse_helpers::hpuStream_t& hpu_stream) {
  auto& time_event_handle_cache = device_->get_time_event_handle_cache();
  if (time_event_handle_cache.get_total_events_count() <
      synapse_helpers::event_handle_cache::get_num_events_high_watermark()) {
    return std::make_shared<synapse_helpers::TimeSlot>(
        device_->get_cached_time_event_handle(),
        device_->get_cached_time_event_handle(),
        static_cast<synStreamHandle>(device_->get_stream(hpu_stream)));
  } else {
    PT_BRIDGE_WARN(
        "High water mark for synapse events ",
        synapse_helpers::event_handle_cache::get_num_events_high_watermark(),
        " reached, will not create any time event");
    return nullptr;
  }
}

} // namespace habana
