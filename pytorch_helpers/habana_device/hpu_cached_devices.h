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

#include <c10/util/Exception.h>
#include <synapse_api_types.h>
#include <synapse_helpers/device.h>

#include <array>
#include <memory>
#include <thread>

namespace synapse_helpers {
class HPURegistrar {
  HPURegistrar() = default;
  std::array<std::shared_ptr<synapse_helpers::device>, MAX_DEVICES_PER_BOX>
      acquired_devices;
  static HPURegistrar& get_hpu_registrar();

 public:
  HPURegistrar(HPURegistrar const&) = delete;
  void operator=(HPURegistrar const&) = delete;

  // This function always return initialized device
  static synapse_helpers::device& get_device(int device_id) {
    auto ret = get_hpu_registrar().acquired_devices.at(device_id).get();
    TORCH_CHECK(ret != nullptr, "Device ", device_id, "is not initialized");
    return *ret;
  }

  static synapse_helpers::device& get_device() {
    const auto& end = get_hpu_registrar().acquired_devices.end();
    auto ret = std::find_if(
        get_hpu_registrar().acquired_devices.begin(),
        end,
        [](auto& device_ptr) { return device_ptr.get() != nullptr; });

    TORCH_CHECK(ret != end, "Habana device not initialized");
    return *(ret->get());
  }

  static void insert_device(std::shared_ptr<synapse_helpers::device> device) {
    get_hpu_registrar().acquired_devices[device->id()] = device;
  }

  static bool empty() {
    for (auto& x : get_hpu_registrar().acquired_devices)
      if (x.get() != nullptr)
        return false;

    return true;
  }

  static bool isInitialized() {
    return initialized_;
  }
  static void markInitialized() {
    initialized_ = true;
  }

  // Delete the acquired device and reset the acquired_devices
  static void deleteDevices() {
    if (initialized_) {
      auto& device = get_hpu_registrar().get_device();
      // Cleanup the device
      device.cleanup();
      // Reset acquired_devices
      get_hpu_registrar().acquired_devices[0] = nullptr;
    }
  }

  static void synchronize_device() {
    auto& device = get_hpu_registrar().get_device();
    device.synchronize();
  }

  static int get_total_device_count() {
    return synapse_helpers::device::get_total_device_count();
  }

  static const std::thread::id& getMainThreadId() {
    return main_thread_id_;
  }

 private:
  static bool initialized_;
  // Note the main thread id
  static const std::thread::id main_thread_id_;
};

} // namespace synapse_helpers
