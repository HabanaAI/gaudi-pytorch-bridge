/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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

#include <c10/util/Exception.h>
#include <synapse_api_types.h>
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/dynamic_shape_info.h"

#include <array>
#include <memory>
#include <thread>

#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/synapse_helpers/device.h"

namespace synapse_helpers {
class HPURegistrar {
  HPURegistrar() = default;
  std::array<std::shared_ptr<synapse_helpers::device>, MAX_DEVICES_PER_BOX>
      acquired_devices;
  static HPURegistrar& get_hpu_registrar();
  ~HPURegistrar() {
    deleteDevices();
    habana::HPUDeviceAllocator::allocator_active_device_id = -1;
    habana::PinnedMemoryAllocator::allocator_active_device_id = -1;
  }

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
      initialized_ = false;
    }
  }

  // Note: Need to finish execution all performed operations till this point
  // Ensure a synchronous mark_step is invoked before calling this function for
  // device synchronization
  static void synchronize_device() {
    auto& device = get_hpu_registrar().get_device();
    device.synchronize();
  }

  static std::string get_device_capability() {
    auto& device = get_hpu_registrar().get_device();
    return device.get_device_capability();
  }

  static std::string get_device_properties(int id) {
    auto& device = get_hpu_registrar().get_device();
    return device.get_device_properties(id);
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
