/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/synapse_helpers/device.h"

namespace synapse_helpers {
class TimeSlot;
}

namespace habana {
namespace backend {
class ScalarCache;
}

class HPUDevice {
 public:
  HPUDevice();
  ~HPUDevice();
  HPUDevice& operator=(const HPUDevice&) = delete;
  HPUDevice& operator=(HPUDevice&&) = delete;
  HPUDevice(const HPUDevice&) = delete;
  HPUDevice(HPUDevice&&) = delete;

  void synchronize() {
    device_->synchronize();
  }

  c10::Device aten_device() {
    return {at::kHPU, static_cast<at::DeviceIndex>(device_->id())};
  }

  synapse_helpers::device_memory& get_device_memory() {
    return device_->get_device_memory();
  }

  synapse_helpers::host_memory& get_host_memory() {
    return device_->get_host_memory();
  }

  synapse_helpers::recipe_handle_cache& get_recipe_handle_cache() {
    return device_->get_recipe_handle_cache();
  }

  bool IsStreamASyncEnabled() const {
    return device_->IsStreamASyncEnabled();
  }

  backend::ScalarCache& GetScalarCache() {
    return device_->get_global_context().GetScalarCache();
  }

  synDeviceId id() const {
    return device_->id();
  }

  std::string name() const {
    return device_->name();
  }

  synDeviceType type() const {
    return device_->type();
  }

  void cleanup() {
    device_->cleanup();
  }

  synapse_helpers::device& syn_device() {
    return *device_;
  }

  using device_ptr = synapse_helpers::device_ptr;
  using hpuStream_t = synapse_helpers::hpuStream_t;

  void copy_data_to_device(
      void* cpu_data,
      device_ptr destination,
      device_ptr event_addr,
      size_t total_bytes,
      const synapse_helpers::event_done_callback& done_cb,
      bool non_blocking = false,
      bool is_pinned = false,
      hpuStream_t hpu_stream = 0) {
    auto syn_error{device_->copy_data_to_device(
        cpu_data,
        destination,
        event_addr,
        total_bytes,
        done_cb,
        non_blocking,
        is_pinned,
        hpu_stream)};
    TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
  }

  void copy_data_to_device(
      synapse_helpers::device::transfer_manifest const& transfers,
      synapse_helpers::event_done_callback unref_cb,
      hpuStream_t hpu_stream = 0) {
    auto syn_error{
        device_->copy_data_to_device(transfers, unref_cb, hpu_stream)};
    TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
  }

  void copy_data_to_host(
      device_ptr device_data,
      void* destination,
      device_ptr event_addr,
      size_t total_bytes,
      const synapse_helpers::event_done_callback& done_cb,
      bool is_pinned = false,
      hpuStream_t hpu_stream = 0) {
    auto syn_error{device_->copy_data_to_host(
        device_data,
        destination,
        event_addr,
        total_bytes,
        done_cb,
        is_pinned,
        hpu_stream)};
    TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
  }

  void copy_data_within_device(
      device_ptr source,
      device_ptr destination,
      device_ptr src_event_addr,
      device_ptr dst_event_addr,
      size_t total_bytes,
      synapse_helpers::event_done_callback unref_cb,
      hpuStream_t hpu_stream = 0) {
    auto syn_error{device_->copy_data_within_device(
        source,
        destination,
        src_event_addr,
        dst_event_addr,
        total_bytes,
        unref_cb,
        hpu_stream)};
    TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
  }

  std::shared_ptr<synapse_helpers::TimeSlot> create_time_slot(
      synapse_helpers::hpuStream_t& hpu_stream);

  int get_count_by_current_type() const {
    return device_->get_count_by_current_type();
  }

 private:
  bool is_stream_async_enabled_;
  synapse_helpers::device_handle device_{nullptr};
};

} // namespace habana
