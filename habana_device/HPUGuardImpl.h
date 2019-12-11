#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include "habana_device/HPUAllocator.h"
#include "habana_device/HPUCheck.h"

#include "synapse/include/synapse_api.h"

#include <unordered_set>

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
    }
    return old_device;
  }
  Device getDevice() const override {
    std::unique_lock<std::mutex> lock(device_lock);
    if (acquired_devices.size() == 0) {
      if (synapse_init == false) {
        TORCH_HABANA_CHECK(synInitialize());
        synapse_init = true;
      }

      // TODO: we are leaking this device, our architecture is not suitable for
      // guard impl
      TORCH_HABANA_CHECK(
          synDeviceAcquireByDeviceType(
              &habana::allocator_active_device_id,
              synDeviceType::synDeviceGaudi),
          "Device acquire failed");
      acquired_devices.emplace(habana::allocator_active_device_id);

      TORCH_HABANA_CHECK(synConfigurationSet("GAUDI_ADDRESS_PATCHING", "true"));
    } else if (acquired_devices.size() == 1)
      habana::allocator_active_device_id = *acquired_devices.begin();
    else
      TORCH_CHECK(
          acquired_devices.size(),
          "Num of acquired devices: ",
          acquired_devices.size(),
          " != 1");

    return Device(DeviceType::HABANA, habana::allocator_active_device_id);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    std::unique_lock<std::mutex> lock(device_lock);
    TORCH_CHECK(
        acquired_devices.find(d.index()) != acquired_devices.end(),
        "device you want to use wasn't acquired");
    habana::allocator_active_device_id = d.index();
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    habana::allocator_active_device_id = d.index();
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

 private:
  // all methods are declared foo() const, so I workaround
  // it with mutable
  mutable std::unordered_set<synDeviceId> acquired_devices;
  mutable std::mutex device_lock;
};

} // namespace detail
} // namespace at
