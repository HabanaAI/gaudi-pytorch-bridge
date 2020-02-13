/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <synapse/include/synapse_api.h>
#include <unordered_set>

#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "habana_helpers/unused_macro.h"
#include "hpu_cached_devices.h"

namespace at {
namespace detail {

struct HABANAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  HABANAGuardImpl() = default;
  DeviceType type() const override {
    return DeviceType::HABANA;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      habana::allocator_active_device_id = d.index();
      TORCH_CHECK(
          habana::allocator_active_device_id == 0,
          "habana active device: ",
          habana::allocator_active_device_id,
          " != 0");
    }
    return old_device;
  }
  Device getDevice() const override {
    if (synapse_helpers::HPURegistrar::empty()) {
      auto device_ptr_or_error = synapse_helpers::device::get_or_create(
          synDeviceType::synDeviceGaudi,
          [](synDeviceId id)
              -> std::unique_ptr<synapse_helpers::device_allocator> {
            return std::make_unique<habana_helpers::HabanaAllocator>(id);
          });

      if (absl::holds_alternative<synapse_helpers::synapse_error>(
              device_ptr_or_error)) {
        auto error =
            absl::get<synapse_helpers::synapse_error>(device_ptr_or_error);
        TORCH_HABANA_CHECK(error.status, error.error);
      } else {
        auto device_ptr = absl::get<std::shared_ptr<synapse_helpers::device>>(
            device_ptr_or_error);
        synapse_helpers::HPURegistrar::insert_device(device_ptr);
      }
    }
    auto& device = synapse_helpers::HPURegistrar::get_device();
    habana::allocator_active_device_id = device.id();

    TORCH_CHECK(
        habana::allocator_active_device_id == 0,
        "habana active device: ",
        habana::allocator_active_device_id,
        " != 0");
    return Device(DeviceType::HABANA, habana::allocator_active_device_id);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    habana::allocator_active_device_id =
        synapse_helpers::HPURegistrar::get_device(d.index()).id();
    TORCH_CHECK(
        habana::allocator_active_device_id == 0,
        "habana active device: ",
        habana::allocator_active_device_id,
        " != 0");
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    habana::allocator_active_device_id = d.index();
    TORCH_CHECK(
        habana::allocator_active_device_id == 0,
        "habana active device: ",
        habana::allocator_active_device_id,
        " != 0");
  }
  Stream getStream(UNUSED Device d) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::HABANA, -1));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(UNUSED Stream s) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::HABANA, -1));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // Event-related functions
  void record(
      UNUSED void** event,
      UNUSED const Stream& stream,
      UNUSED const DeviceIndex device_index,
      UNUSED const EventFlag flag) const override {
    TORCH_CHECK(false, "HABANA backend doesn't support events.");
  }
  void block(UNUSED void* event, UNUSED const Stream& stream) const override {
    TORCH_CHECK(false, "HABANA backend doesn't support events.")
  }
  bool queryEvent(UNUSED void* event) const override {
    TORCH_CHECK(false, "HABANA backend doesn't support events.")
  }
  void destroyEvent(UNUSED void* event, UNUSED const DeviceIndex device_index)
      const noexcept override {}
}; // namespace detail

} // namespace detail
} // namespace at
