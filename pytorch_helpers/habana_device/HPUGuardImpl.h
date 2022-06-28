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

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <synapse_api.h>
#include <unordered_set>

#include "HPUAllocator.h"
#include "HPUCheck.h"
#include "HPUStream.h"
#include "PinnedMemoryAllocator.h"
#include "habana_helpers/unused_macro.h"
#include "hpu_cached_devices.h"

using namespace c10::hpu;

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
    return getCurrentHPUStream(d.index()).unwrap();
  }

  at::Stream getDefaultStream(at::Device d) const override {
    return getDefaultHPUStream(d.index());
  }

  at::Stream getStreamFromGlobalPool(at::Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }
  at::Stream exchangeStream(at::Stream s) const noexcept override {
    HPUStream hs(s);
    auto old_stream = getCurrentHPUStream(s.device().index());
    setCurrentHPUStream(hs);
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

  void createEvent(synEventHandle& handle, const at::EventFlag flag) const {
    auto& dev = synapse_helpers::HPURegistrar::get_device();
    unsigned int hpu_flag = get_hpu_flag(flag);
    if (hpu_flag) {
      handle = dev.get_time_event_handle_cache().get_free_handle();
    } else {
      handle = dev.get_event_handle_cache().get_free_handle();
    }
    dev.add_user_event(handle, hpu_flag);
  }

  void destroyEvent(void* event, UNUSED const at::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    synEventHandle handle = static_cast<synEventHandle>(event);
    if (handle) {
      PT_DEVICE_DEBUG("Event:: removing the handle", handle);
      auto& dev = synapse_helpers::HPURegistrar::get_device();
      if (dev.get_user_event_flag(handle)) {
        dev.get_time_event_handle_cache().release_handle(handle);
      } else {
        dev.get_event_handle_cache().release_handle(handle);
      }
      dev.remove_user_event(handle);
    }
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
    synEventHandle handle = static_cast<synEventHandle>(*event);
    HPUStream hpu_stream{stream};

    // Creates the event (lazily)
    if (!handle) {
      synEventHandle syn_handle{};
      createEvent(syn_handle, flag);
      handle = syn_handle;
    }

    *event = handle;
    if (stream == c10::hpu::getCurrentHPUStream()) {
      bool async =
          (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD) &&
           GET_ENV_FLAG_NEW(PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD));
      habana_lazy::HbLazyTensor::StepMarker(
          {},
          nullptr,
          {},
          async,
          handle,
          hpu_stream.stream(),
          get_hpu_flag(flag));
    } else {
      auto& device = synapse_helpers::HPURegistrar::get_device();
      auto status = synEventRecord(
          handle, device.get_compute_stream(hpu_stream.stream()));
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL("synEventRecord failed ", status);
      }
    }
  }

  void block(void* event, const at::Stream& stream) const override {
    if (!event)
      return;
    synEventHandle handle = static_cast<synEventHandle>(event);
    habana_lazy::HbLazyTensor::StepMarkerFinish();
    HPUStream hpu_stream{stream};
    auto& device = synapse_helpers::HPURegistrar::get_device();
    auto status = synStreamWaitEvent(
        device.get_compute_stream(hpu_stream.stream()), handle, 0);
    if (synStatus::synSuccess != status) {
      PT_DEVICE_FATAL("synStreamWaitEvent failed: ", status);
    }
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    synEventHandle handle = static_cast<synEventHandle>(event);
    auto status = synEventQuery(handle);
    if (status == synSuccess) {
      return true;
    } else {
      PT_DEVICE_DEBUG("STREAM:: synEventQuery failed with status", status);
    }

    return false;
  }

  // Stream-related functions
  bool queryStream(const at::Stream& stream) const override {
    HPUStream hpu_stream{stream};
    return hpu_stream.query();
  }

  void synchronizeStream(const at::Stream& stream) const override {
    HPUStream hpu_stream{stream};
    hpu_stream.synchronize();
  }
};
} // namespace habana
