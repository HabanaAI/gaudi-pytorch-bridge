#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include "habana_device/HPUCheck.h"

#include "synapse/include/synapse_api.h"

#include <unordered_set>

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
      active_device = d.index();
    }
    return old_device;
  }
  Device getDevice() const override {
    std::unique_lock<std::mutex> lock(device_lock);
    if (acquired_devices.size() == 0) {
      // TODO: we are leaking this device, our architecture is not suitable for
      // guard impl
      auto status = synDeviceAcquireByDeviceType(
          &active_device, synDeviceType::synDeviceGaudi);
      TORCH_HABANA_CHECK(status, "Device acquire failed");
      acquired_devices.emplace(active_device);
    } else if (acquired_devices.size() == 1)
      active_device = *acquired_devices.begin();
    else
      TORCH_CHECK(
          acquired_devices.size(),
          "Num of acquired devices: ",
          acquired_devices.size(),
          " != 1");

    return Device(DeviceType::HABANA, active_device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    std::unique_lock<std::mutex> lock(device_lock);
    TORCH_CHECK(
        acquired_devices.find(d.index()) != acquired_devices.end(),
        "device you want to use wasn't acquired");
    active_device = d.index();
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    active_device = d.index();
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
  mutable synDeviceId active_device = -1;
  mutable std::mutex device_lock;
};

} // namespace detail
} // namespace at
