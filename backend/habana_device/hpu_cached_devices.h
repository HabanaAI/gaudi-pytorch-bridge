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

#include <array>
#include <memory>
#include <mutex>
#include <thread>

#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/helpers/dynamic_shape_info.h"
#include "backend/synapse_helpers/device.h"

namespace habana {
class HPURegistrar {
  HPURegistrar() = default;
  std::array<synapse_helpers::device_handle, MAX_DEVICES_PER_BOX>
      acquired_devices;
  static HPURegistrar& get_hpu_registrar();
  ~HPURegistrar() {
    deleteDevices();
    HPUDeviceAllocator::allocator_active_device_id = -1;
    PinnedMemoryAllocator::allocator_active_device_id = -1;
  }

 public:
  HPURegistrar(HPURegistrar const&) = delete;
  void operator=(HPURegistrar const&) = delete;

  class HPUGlobalConfig {
   public:
    HPUGlobalConfig() = default;

    bool getDeterministic() {
      std::lock_guard<std::mutex> lock(config_lock_);
      return deterministic_;
    }
    void setDeterministic(bool val) {
      std::lock_guard<std::mutex> lock(config_lock_);
      deterministic_ = val;
    }

   private:
    /* TO DO
       ENV flag PT_HPU_ENABLE_DETERMINISTIC_MODE will be removed
       after model script migration to deterministic API.

       bool deterministic_{false};
    */
    bool deterministic_ = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DETERMINISTIC_MODE);
    std::mutex config_lock_{};
  };

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

  static void insert_device(synapse_helpers::device_handle device) {
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

  static HPUGlobalConfig& get_hpu_global_config() {
    static HPUGlobalConfig instance_;
    return instance_;
  }

 private:
  static bool initialized_;
  // Note the main thread id
  static const std::thread::id main_thread_id_;
};

} // namespace habana
