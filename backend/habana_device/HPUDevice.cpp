/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/scalar_cache.h"
#include "backend/synapse_helpers/time_slot.h"
#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"

namespace habana {

HPUDevice::HPUDevice()
    : scalar_cache_{std::make_unique<backend::ScalarCache>()} {
  auto device_ptr_or_error = synapse_helpers::device::get_or_create(
      synapse_helpers::device::get_supported_devices());

  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          device_ptr_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(device_ptr_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  } else {
    auto device_ptr =
        absl::get<synapse_helpers::device_handle>(device_ptr_or_error);
    device_ = std::move(device_ptr);
  }
}

HPUDevice::~HPUDevice() {
  /* In this scope the runtime is finalizing and there should be no more
   * references to the synapse device beyond the one owned by the HPUDevice.
   * Presence of other references should be investigated since it might not be
   * safe to finalize the synapse device later.
   **/
  constant_information->ClearChecksumInformation();
  if (device_.use_count() != 1) {
    TORCH_WARN(
        "when deleting HPUDevice, device is kept alive by ",
        device_.use_count() - 1,
        " other references ");
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

habana_helpers::ThreadPool& HPUDevice::create_lowering_thread() {
  class ThreadPoolHolder final : public DeviceResource {
   public:
    habana_helpers::ThreadPool thread_pool_{true};
  };
  auto lowering_thread{std::make_unique<ThreadPoolHolder>()};
  raw_lowering_thread_ = &lowering_thread->thread_pool_;
  lowering_thread_.reset(lowering_thread.release());
  return *raw_lowering_thread_;
}

} // namespace habana
