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

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <synapse_api.h>
#include <unordered_set>

#include "HPUAllocator.h"
#include "HPUEvent.h"
#include "HPUStream.h"
#include "PinnedMemoryAllocator.h"
#include "habana_helpers/logging.h"
#include "hpu_cached_devices.h"

namespace habana {
struct HABANAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr at::DeviceType static_devType = at::DeviceType::HPU;

  HABANAGuardImpl() = default;
  at::DeviceType type() const override {
    return at::DeviceType::HPU;
  }
  at::Device exchangeDevice(at::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == type());
    at::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      HPUDeviceAllocator::allocator_active_device_id = d.index();
      TORCH_CHECK(
          habana::HPUDeviceAllocator::allocator_active_device_id == 0,
          "habana active device: ",
          habana::HPUDeviceAllocator::allocator_active_device_id,
          " != 0");
    }
    return old_device;
  }

  at::Device getDevice() const override {
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
        return std::make_unique<habana::HPUAllocator>(id);
      };
      // Create the synapse_helpers::device, which will create the OSAL object.
      auto device_ptr_or_error = synapse_helpers::device::get_or_create(
          synapse_helpers::device::get_supported_devices(), allocatorVar);

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
    return at::Device(
        at::DeviceType::HPU,
        habana::HPUDeviceAllocator::allocator_active_device_id);
  }
  void setDevice(at::Device d) const override {
    // For CPU device, fork is invoking set_device from Engine::thread_init with
    // device=0. Hence, as the HPU device won't be initialized at that point,
    // silently ignore the call.

    // NOTE: Current setDevice() implementation is a non-functioning one.
    // There is no runtime update for any setDevice call.
    // As there is always 1 device in play all the time,
    // setDevice() usage wont be required currently.

    if (synapse_helpers::HPURegistrar::isInitialized()) {
      TORCH_INTERNAL_ASSERT(d.type() == type());
      habana::HPUDeviceAllocator::allocator_active_device_id =
          synapse_helpers::HPURegistrar::get_device(d.index()).id();
      TORCH_CHECK(
          habana::HPUDeviceAllocator::allocator_active_device_id == 0,
          "habana active device: ",
          habana::HPUDeviceAllocator::allocator_active_device_id,
          " != 0");
    }
  }

  void uncheckedSetDevice(at::Device d) const noexcept override {
    habana::HPUDeviceAllocator::allocator_active_device_id = d.index();
    if (habana::HPUDeviceAllocator::allocator_active_device_id != 0)
      TORCH_WARN(
          "habana active device: ",
          habana::HPUDeviceAllocator::allocator_active_device_id,
          " != 0");
  }
  at::Stream getStream(at::Device d) const noexcept override {
    return c10::hpu::getCurrentHPUStream(d.index()).unwrap();
  }

  at::Stream getDefaultStream(at::Device d) const override {
    return c10::hpu::getDefaultHPUStream(d.index());
  }

  at::Stream getStreamFromGlobalPool(at::Device d, bool isHighPriority = false)
      const override {
    return c10::hpu::getStreamFromPool(isHighPriority, d.index());
  }
  at::Stream exchangeStream(at::Stream s) const noexcept override {
    c10::hpu::HPUStream hs(s);
    auto old_stream = c10::hpu::getCurrentHPUStream(s.device().index());
    c10::hpu::setCurrentHPUStream(hs);
    return old_stream.unwrap();
  }

  at::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // Event-related functions
  static unsigned int get_hpu_flag(const at::EventFlag flag) {
    // Maps PyTorch's Event::Flag to HPU flag
    unsigned int hpu_flag = 1; // Enable timing
    switch (flag) {
      case at::EventFlag::PYTORCH_DEFAULT:
        hpu_flag = 0;
        break;
      case at::EventFlag::BACKEND_DEFAULT:
        hpu_flag = 1;
        break;
      default:
        TORCH_CHECK(false, "event received unknown flag");
    }
    return hpu_flag;
  }

  void destroyEvent(
      void* event,
      [[maybe_unused]] const at::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;

    at::hpu::HPUEvent* hpu_event = static_cast<at::hpu::HPUEvent*>(event);
    delete hpu_event;
  }

  void record(
      void** event,
      const at::Stream& stream,
      const at::DeviceIndex device_index,
      const at::EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");
    at::hpu::HPUEvent* hpu_event = (static_cast<at::hpu::HPUEvent*>(*event));
    c10::hpu::HPUStream hpu_stream{stream};
    if (!hpu_event) {
      unsigned int hpu_flag = get_hpu_flag(flag);
      hpu_event = new at::hpu::HPUEvent(hpu_flag);
    }
    hpu_event->record(hpu_stream);
    *event = (void*)hpu_event;
  }

  void block(void* event, const at::Stream& stream) const override {
    if (!event)
      return;
    at::hpu::HPUEvent* hpu_event = static_cast<at::hpu::HPUEvent*>(event);
    c10::hpu::HPUStream hpu_stream{stream};
    hpu_event->block(hpu_stream);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    at::hpu::HPUEvent* hpu_event = static_cast<at::hpu::HPUEvent*>(event);
    return hpu_event->query();
  }

  // Stream-related functions
  bool queryStream(const at::Stream& stream) const override {
    c10::hpu::HPUStream hpu_stream{stream};
    return hpu_stream.query();
  }

  void synchronizeStream(const at::Stream& stream) const override {
    c10::hpu::HPUStream hpu_stream{stream};
    hpu_stream.synchronize();
  }
};
} // namespace habana
