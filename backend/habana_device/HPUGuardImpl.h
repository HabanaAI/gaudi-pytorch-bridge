/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
#include "HPUDevice.h"
#include "HPUEvent.h"
#include "HPUStream.h"
#include "PinnedMemoryAllocator.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/lazy_executor.h"
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
    return HPUDeviceContext::get_or_create_aten_device();
  }

  void setDevice(at::Device d) const override {
    // For CPU device, fork is invoking set_device from Engine::thread_init with
    // device=0. Hence, as the HPU device won't be initialized at that point,
    // silently ignore the call.

    // NOTE: Current setDevice() implementation is a non-functioning one.
    // There is no runtime update for any setDevice call.
    // As there is always 1 device in play all the time,
    // setDevice() usage wont be required currently.

    if (HPUDeviceContext::is_device_acquired()) {
      TORCH_INTERNAL_ASSERT(d.type() == type());
      habana::HPUDeviceAllocator::allocator_active_device_id =
          HPUDeviceContext::get_device(d.index()).id();
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

  at::Stream getNewStream(at::Device d, int priority = 0) const override {
    return c10::hpu::getStreamFromPool(priority, d.index());
  }

  at::Stream exchangeStream(at::Stream s) const noexcept override {
    c10::hpu::HPUStream hs(s);
    auto old_stream = c10::hpu::getCurrentHPUStream(s.device().index());
    c10::hpu::setCurrentHPUStream(hs);
    return old_stream.unwrap();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    auto* hpu_event = static_cast<at::hpu::HPUEvent*>(event);
    hpu_event->synchronize();
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

  double elapsedTime(
      void* event1,
      void* event2,
      C10_UNUSED const c10::DeviceIndex device_index) const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    auto* hpu_event1 = static_cast<at::hpu::HPUEvent*>(event1);
    auto* hpu_event2 = static_cast<at::hpu::HPUEvent*>(event2);
    float time_ms = hpu_event1->elapsed_time(*hpu_event2) / 1e6;
    return static_cast<double>(time_ms);
  }
};
} // namespace habana
