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
#include <synapse_api.h>
#include <unordered_set>

#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "PinnedMemoryAllocator.h"
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
      habana::HPUDeviceAllocator::allocator_active_device_id = d.index();
      TORCH_CHECK(
          habana::HPUDeviceAllocator::allocator_active_device_id == 0,
          "habana active device: ",
          habana::HPUDeviceAllocator::allocator_active_device_id,
          " != 0");
    }
    return old_device;
  }

  Device getDevice() const override {
    /**
       NOTE: From https://en.cppreference.com/w/cpp/utility/program/atexit
         The functions may be called concurrently with the destruction of the
     objects with static storage duration and with each other, maintaining the
     guarantee that if registration of A was sequenced-before the registration
     of B, then the call to B is sequenced-before the call to A, same applies to
     the sequencing between static object constructors and calls to atexit: see
     std::exit

     Static creation order:
       When synapse_helpers::HPURegistrar::empty() is called to check if a
     device is created already, the HPRegistrar object is created before OSAL.
       The function at::detail::HABANAGuardImpl::getDevice calls this getDevice.
       When this method first calls synapse_helpers::HPURegistrar::empty(), it
     creates the HPRegistrar object. Later, when the HABANAGuardImpl::getDevice
     calls synapse_helpers::device::get_or_create, the OSAL object is created.
     Static destruction order:
      Since HPRegistrar is created before OSAL, thedestruction order is ~OSAL
     followed by ~HPURegistrar, as per the NOTE above.

     Methods to resolve:
      1> Using C++ atexit handler to clean the devices
         The C++ exit handler doesn't help, as they run too late and the OSAL in
     synapse is already destroyed before we reach here - this results in the
     device stream destruction to fail within OSAL code. Essentially, this
     follows the order mentioned in NOTE above. 2> Use python atexit handler
        PyTorch atexit handler executes when the Pythin interpreter exits. This
     cleans up the device properly. However, the PyTorch imported modules still
     gets unloaded after this in the exit path, and tensors held by the
     framework gets released at this point. These tensor releases fail as the
     storage destruction depend on the device to be available. 3> Change the
     creation order of OSAL and HPRegistrar Create the synapse_helpers::device
     before creatng the HPRegistrar object. For this to work, use a static flag
     to check whether the device have been created already or not. If the device
     isn't created, then create it first, resulting in OSAL object creation.
        After this, create the HPURegistrar object and insert the device.
        This ensures that the destruction order is ~HPRegistrar followed by
     ~OSAL, which ensures correct destruction of devices.
    */
    if (!synapse_helpers::HPURegistrar::isInitialized()) {
      auto allocatorVar = [](synDeviceId id)
          -> std::unique_ptr<synapse_helpers::device_allocator> {
        return std::make_unique<at::habana::HPUAllocator>(id);
      };
      // Create the synapse_helpers::device, which will create the OSAL object.
      auto device_ptr_or_error = synapse_helpers::device::get_or_create(
          {synDeviceType::synDeviceGaudi, synDeviceType::synDeviceGaudiM},
          allocatorVar);

      if (absl::holds_alternative<synapse_helpers::synapse_error>(
              device_ptr_or_error)) {
        auto error =
            absl::get<synapse_helpers::synapse_error>(device_ptr_or_error);
        TORCH_HABANA_CHECK(error.status, error.error);
      } else {
        auto device_ptr = absl::get<std::shared_ptr<synapse_helpers::device>>(
            device_ptr_or_error);
        // Insert the device in HPURegistrar. The
        // synapse_helpers::HPURegistrar::empty() call creates the HPURegistrar
        // object.
        TORCH_CHECK(
            synapse_helpers::HPURegistrar::empty(),
            "HPURegistrar not empty when synapse device is being created");
        synapse_helpers::HPURegistrar::insert_device(device_ptr);
        // Mark the HPURegistrar to be initialized with a device
        synapse_helpers::HPURegistrar::markInitialized();
      }
    }
    auto& device = synapse_helpers::HPURegistrar::get_device();
    habana::HPUDeviceAllocator::allocator_active_device_id = device.id();
    habana::PinnedMemoryAllocator::allocator_active_device_id = device.id();

    TORCH_CHECK(
        habana::HPUDeviceAllocator::allocator_active_device_id == 0,
        "habana active device: ",
        habana::HPUDeviceAllocator::allocator_active_device_id,
        " != 0");
    return Device(
        DeviceType::HABANA,
        habana::HPUDeviceAllocator::allocator_active_device_id);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    habana::HPUDeviceAllocator::allocator_active_device_id =
        synapse_helpers::HPURegistrar::get_device(d.index()).id();
    TORCH_CHECK(
        habana::HPUDeviceAllocator::allocator_active_device_id == 0,
        "habana active device: ",
        habana::HPUDeviceAllocator::allocator_active_device_id,
        " != 0");
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    habana::HPUDeviceAllocator::allocator_active_device_id = d.index();
    if (habana::HPUDeviceAllocator::allocator_active_device_id != 0)
      TORCH_WARN(
          "habana active device: ",
          habana::HPUDeviceAllocator::allocator_active_device_id,
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
