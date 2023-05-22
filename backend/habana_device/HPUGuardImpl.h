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
    return hpu_registrar().get_or_create_device().aten_device();
  }

  void setDevice(at::Device d) const override {
    // For CPU device, fork is invoking set_device from Engine::thread_init with
    // device=0. Hence, as the HPU device won't be initialized at that point,
    // silently ignore the call.

    // NOTE: Current setDevice() implementation is a non-functioning one.
    // There is no runtime update for any setDevice call.
    // As there is always 1 device in play all the time,
    // setDevice() usage wont be required currently.

    if (HPURegistrar::get_hpu_registrar().is_initialized()) {
      TORCH_INTERNAL_ASSERT(d.type() == type());
      habana::HPUDeviceAllocator::allocator_active_device_id =
          HPURegistrar::get_device(d.index()).id();
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

  void createEvent(
      synEventHandle& handle,
      const at::EventFlag flag,
      synapse_helpers::hpuStream_t stream) const {
    auto& dev = HPURegistrar::get_device().syn_device();
    unsigned int hpu_flag = get_hpu_flag(flag);
    if (hpu_flag) {
      handle = dev.get_time_event_handle_cache().get_free_handle();
    } else {
      handle = dev.get_event_handle_cache().get_free_handle();
    }
    dev.add_user_event_info(handle, hpu_flag, stream);
  }

  void destroyEvent(
      void* event,
      [[maybe_unused]] const at::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    synEventHandle handle = static_cast<synEventHandle>(event);
    if (handle) {
      PT_DEVICE_DEBUG("Event:: removing the handle", handle);
      auto& dev = HPURegistrar::get_device().syn_device();
      std::pair<bool, synapse_helpers::hpuStream_t> event_info =
          dev.get_user_event_info(handle);
      if (event_info.second == 0 &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) {
        dev.delete_event_default_stream(handle, event_info.first);
      } else {
        if (event_info.first) {
          dev.get_time_event_handle_cache().release_handle(handle);
        } else {
          dev.get_event_handle_cache().release_handle(handle);
        }
      }
      dev.remove_user_event_info(handle);
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
    c10::hpu::HPUStream hpu_stream{stream};
    auto& device = HPURegistrar::get_device().syn_device();

    // Creates the event (lazily)
    if (!handle) {
      synEventHandle syn_handle{};
      createEvent(syn_handle, flag, hpu_stream.stream());
      if (hpu_stream.stream() == 0 &&
          GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
        unsigned int hpu_flag = get_hpu_flag(flag);
        device.create_default_stream_event(handle, hpu_flag);
      }
      handle = syn_handle;
    }

    *event = handle;
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1 &&
        stream == c10::hpu::getCurrentHPUStream()) {
      habana_lazy::HbLazyTensor::StepMarker({});
    }

    if (hpu_stream.stream() == 0 &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
      device.record_event_default_stream(handle);
    } else {
      auto status =
          synEventRecord(handle, device.get_stream(hpu_stream.stream()));
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL(
            Logger::formatStatusMsg(status), "synEventRecord failed");
      }
    }
  }

  void block(void* event, const at::Stream& stream) const override {
    if (!event)
      return;
    synEventHandle handle = static_cast<synEventHandle>(event);
    c10::hpu::HPUStream hpu_stream{stream};
    auto& device = HPURegistrar::get_device().syn_device();
    std::pair<bool, synapse_helpers::hpuStream_t> event_info =
        device.get_user_event_info(handle);
    if (event_info.second == hpu_stream.stream())
      return;

    if (hpu_stream.stream() == 0 &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
      device.wait_event_default_stream(handle);
    } else {
      auto status =
          synStreamWaitEvent(device.get_stream(hpu_stream.stream()), handle, 0);
      if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL(
            Logger::formatStatusMsg(status), "synStreamWaitEvent failed");
      }
    }
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    synEventHandle handle = static_cast<synEventHandle>(event);
    auto& device = HPURegistrar::get_device().syn_device();
    std::pair<bool, synapse_helpers::hpuStream_t> event_info =
        device.get_user_event_info(handle);
    if (event_info.second == 0 &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM)) { // default stream
      return device.query_event_default_stream(handle);
    } else {
      auto status = synEventQuery(handle);
      if (status == synSuccess) {
        return true;
      } else {
        PT_DEVICE_DEBUG(
            Logger::formatStatusMsg(status), "STREAM:: synEventQuery");
      }
    }

    return false;
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
