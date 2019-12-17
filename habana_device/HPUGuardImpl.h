#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <synapse/include/synapse_api.h>
#include <unordered_set>

#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "hpu_cached_devices.h"

namespace at {
namespace detail {

extern bool synapse_init;

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
      if (!synapse_init) {
        TORCH_HABANA_CHECK(synInitialize());
        synapse_init = true;
      }

      // TODO: create shouldn't get device id as input, I don't know which
      // device will be acquired
      auto device_ptr_or_error = synapse_helpers::device::create(
          synDeviceType::synDeviceGaudi,
          std::make_unique<habana_helpers::HabanaAllocator>(0));

      if (absl::holds_alternative<synapse_helpers::synapse_error>(
              device_ptr_or_error)) {
        auto error =
            absl::get<synapse_helpers::synapse_error>(device_ptr_or_error);
        TORCH_HABANA_CHECK(error.status, error.error);
      } else {
        auto device_ptr = absl::get<std::unique_ptr<synapse_helpers::device>>(
            std::move(device_ptr_or_error));
        synapse_helpers::HPURegistrar::insert_device(std::move(device_ptr));
      }
    } else {
      // TODO: we are always asking for device 0, it may change in the future
      auto& device = synapse_helpers::HPURegistrar::get_device(0);
      habana::allocator_active_device_id = device.id();
    }

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
  Stream getStream(Device d) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::HABANA, -1));
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    // no-op
    return Stream(Stream::DEFAULT, Device(DeviceType::HABANA, -1));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // Event-related functions
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(false, "HABANA backend doesn't support events.");
  }
  void block(void* event, const Stream& stream) const override {
    TORCH_CHECK(false, "HABANA backend doesn't support events.")
  }
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "HABANA backend doesn't support events.")
  }
  void destroyEvent(void* event, const DeviceIndex device_index) const
      noexcept override {}
}; // namespace detail

} // namespace detail
} // namespace at
