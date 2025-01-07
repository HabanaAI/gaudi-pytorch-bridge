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

#include "backend/habana_device/HPUDevice.h"
#include <memory>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/scalar_cache.h"
#include "backend/synapse_helpers/time_slot.h"
#include "pytorch_helpers/habana_helpers/python_utils.h"

namespace habana {

void ThreadPoolWithGILRelease::waitWorkComplete() {
  // TODO remove gil_release once SW-160978 is fixed
  habana_helpers::AutoNoGIL gil_release;
  habana_helpers::ThreadPool::waitWorkComplete();
}

struct HPUDeviceContextImpl {
  std::unique_ptr<habana_helpers::ThreadPool> garbage_collection_thread_;
  synapse_helpers::device_handle device_;
  std::unique_ptr<backend::ScalarCache> scalar_cache_;

  std::unique_ptr<RecipeCacheLRU> recipe_cache_;

  std::unique_ptr<ThreadPoolWithGILRelease> execute_thread_;
  std::unique_ptr<ThreadPoolWithGILRelease> compile_thread_;
  std::unique_ptr<ThreadPoolWithGILRelease> lowering_thread_;

  // Holding this is required for proper destruction order
  std::shared_ptr<ConstantInformation> constant_information_;
  bool exception_occurred_ = false;
  void Init();
  void JoinAllThreads();
  void JoinPipelineThreads();
  void CreateDevice();
  void Finish();
} device_context;

void HPUDeviceContextImpl::JoinAllThreads() {
  if (!lowering_thread_)
    return;
  JoinPipelineThreads();
  garbage_collection_thread_->waitWorkComplete();
}
void HPUDeviceContextImpl::JoinPipelineThreads() {
  if (!lowering_thread_)
    return;
  try {
    lowering_thread_->waitWorkComplete();
  } catch (...) {
    exception_occurred_ = true;
    throw;
  }

  compile_thread_->waitWorkComplete();
  execute_thread_->waitWorkComplete();
}

synapse_helpers::device& HPUDeviceContext::get_device(int) {
  HABANA_ASSERT(device_context.device_);
  return *device_context.device_;
}

void HPUDeviceContextImpl::CreateDevice() {
  auto device_ptr_or_error = synapse_helpers::device::get_or_create(
      synapse_helpers::device::get_supported_devices());

  if (absl::holds_alternative<synapse_helpers::synapse_error>(
          device_ptr_or_error)) {
    auto error = absl::get<synapse_helpers::synapse_error>(device_ptr_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  } else {
    device_ = absl::get<synapse_helpers::device_handle>(device_ptr_or_error);
  }
}

void HPUDeviceContextImpl::Init() {
  CreateDevice();
  garbage_collection_thread_ =
      std::make_unique<habana_helpers::ThreadPool>(true);
  recipe_cache_ = std::make_unique<RecipeCacheLRU>();
  execute_thread_ = std::make_unique<ThreadPoolWithGILRelease>();
  compile_thread_ = std::make_unique<ThreadPoolWithGILRelease>();
  lowering_thread_ = std::make_unique<ThreadPoolWithGILRelease>();
  constant_information_ = ConstantInformationPtr();
  scalar_cache_ = std::make_unique<backend::ScalarCache>();

  HPURegistrar::get_hpu_registrar().register_device_deleter(
      []() { device_context.Finish(); });
}

void HPUDeviceContextImpl::Finish() {
  {
    habana_helpers::AutoNoGIL gil_release;
    device_context.lowering_thread_.reset();
    device_context.compile_thread_.reset();
    device_context.execute_thread_.reset();
  }
  device_context.recipe_cache_.reset();
  device_context.scalar_cache_.reset();

  device_context.constant_information_->ClearChecksumInformation();

  // We have to remove garbage_collection_thread_ after destroying the stream
  // but before releasing the device_id. Both classes are owned by class device
  // So, we have to use this workaround till we refactor device class by
  // decomposing it into smaller classes
  device_context.device_->cleanup();

  device_context.garbage_collection_thread_.reset();
  device_context.device_.reset();

  if (device_context.device_.use_count() != 0) {
    TORCH_WARN(
        "when deleting HPUDevice, device is kept alive by ",
        device_context.device_.use_count(),
        " other references ");
  }

  habana::HPUDeviceAllocator::allocator_active_device_id = -1;
  habana::PinnedMemoryAllocator::allocator_active_device_id = -1;
  device_context.constant_information_.reset();
}

namespace HPUDeviceContext {

void join_all_threads() {
  device_context.JoinAllThreads();
}
void join_pipeline_threads() {
  device_context.JoinPipelineThreads();
}

bool get_exception_occurred() {
  if (!is_device_acquired())
    return false;
  bool exception_occurred = device_context.exception_occurred_;
  device_context.exception_occurred_ = false;
  return exception_occurred;
}

ThreadPoolWithGILRelease& compile_thread() {
  HABANA_ASSERT(device_context.compile_thread_);
  return *device_context.compile_thread_;
}

habana_helpers::ThreadPool& garbage_collection_thread() {
  HABANA_ASSERT(device_context.garbage_collection_thread_);
  return *device_context.garbage_collection_thread_;
}

ThreadPoolWithGILRelease& lowering_thread() {
  HABANA_ASSERT(device_context.lowering_thread_);
  return *device_context.lowering_thread_;
}

ThreadPoolWithGILRelease& execute_thread() {
  HABANA_ASSERT(device_context.execute_thread_);
  return *device_context.execute_thread_;
}

backend::ScalarCache& scalar_cache() {
  HABANA_ASSERT(device_context.scalar_cache_);
  return *device_context.scalar_cache_;
}

synapse_helpers::device& syn_device() {
  HABANA_ASSERT(device_context.device_);
  return *device_context.device_;
}

RecipeCacheLRU& recipe_cache() {
  HABANA_ASSERT(device_context.recipe_cache_);
  return *device_context.recipe_cache_;
}

void recipe_cache_clear() {
  if (device_context.recipe_cache_)
    device_context.recipe_cache_->clear();
}

void flush_disk_cache() {
  if (device_context.recipe_cache_)
    device_context.recipe_cache_->FlushDiskCache();
}

void synchronize() {
  HABANA_ASSERT(device_context.device_);
  device_context.device_->synchronize();
}

void synchronize_host_multistage_pipeline() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    device_context.JoinAllThreads();
  }
}

std::string get_device_capability() {
  HABANA_ASSERT(device_context.device_);
  return device_context.device_->get_device_capability();
}

std::string get_device_properties(unsigned id) {
  return synapse_helpers::device::get_device_properties(id);
}

int get_total_device_count() {
  return synapse_helpers::device::get_total_device_count();
}

c10::Device get_or_create_aten_device() {
  PT_BRIDGE_BEGIN;
  if (!device_context.device_) {
    device_context.Init();
    PT_BRIDGE_DEBUG("Created hpu device ", device_context.device_.get());
    habana::HPUDeviceAllocator::allocator_active_device_id = 0;
    habana::PinnedMemoryAllocator::allocator_active_device_id = 0;
  }
  return {at::kHPU, static_cast<at::DeviceIndex>(0)};
}

c10::Device aten_device() {
  HABANA_ASSERT(device_context.device_);
  return {at::kHPU, static_cast<at::DeviceIndex>(0)};
}

void copy_data_to_device(
    void* cpu_data,
    synapse_helpers::device_ptr destination,
    synapse_helpers::device_ptr event_addr,
    size_t total_bytes,
    const synapse_helpers::event_done_callback& done_cb,
    bool non_blocking,
    bool is_pinned,
    synapse_helpers::hpuStream_t hpu_stream,
    void* host_cpu_data) {
  HABANA_ASSERT(device_context.device_);
  auto syn_error{device_context.device_->copy_data_to_device(
      cpu_data,
      destination,
      event_addr,
      total_bytes,
      done_cb,
      non_blocking,
      is_pinned,
      hpu_stream,
      host_cpu_data)};
  TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
}

void copy_data_to_device(
    synapse_helpers::device::transfer_manifest const& transfers,
    synapse_helpers::event_done_callback unref_cb,
    synapse_helpers::hpuStream_t hpu_stream) {
  HABANA_ASSERT(device_context.device_);
  auto syn_error{device_context.device_->copy_data_to_device(
      transfers, unref_cb, hpu_stream)};
  TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
}

void copy_data_to_host(
    synapse_helpers::device_ptr device_data,
    void* destination,
    synapse_helpers::device_ptr event_addr,
    size_t total_bytes,
    const synapse_helpers::event_done_callback& done_cb,
    bool is_pinned,
    synapse_helpers::hpuStream_t hpu_stream) {
  HABANA_ASSERT(device_context.device_);
  auto syn_error{device_context.device_->copy_data_to_host(
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
    synapse_helpers::device_ptr source,
    synapse_helpers::device_ptr destination,
    synapse_helpers::device_ptr src_event_addr,
    synapse_helpers::device_ptr dst_event_addr,
    size_t total_bytes,
    synapse_helpers::event_done_callback unref_cb,
    synapse_helpers::hpuStream_t hpu_stream) {
  HABANA_ASSERT(device_context.device_);
  auto syn_error{device_context.device_->copy_data_within_device(
      source,
      destination,
      src_event_addr,
      dst_event_addr,
      total_bytes,
      unref_cb,
      hpu_stream)};
  TORCH_HABANA_CHECK(syn_error.status, syn_error.error);
}

synapse_helpers::device_memory& get_device_memory() {
  HABANA_ASSERT(device_context.device_);
  return device_context.device_->get_device_memory();
}

synapse_helpers::host_memory& get_host_memory() {
  HABANA_ASSERT(device_context.device_);
  return device_context.device_->get_host_memory();
}

std::shared_ptr<synapse_helpers::TimeSlot> create_time_slot(
    synapse_helpers::hpuStream_t& hpu_stream) {
  HABANA_ASSERT(device_context.device_);
  auto& device = *device_context.device_;
  auto& time_event_handle_cache = device.get_time_event_handle_cache();
  if (time_event_handle_cache.get_total_events_count() <
      synapse_helpers::event_handle_cache::get_num_events_high_watermark()) {
    return std::make_shared<synapse_helpers::TimeSlot>(
        device.get_cached_time_event_handle(),
        device.get_cached_time_event_handle(),
        static_cast<synStreamHandle>(device.get_stream(hpu_stream)));
  } else {
    PT_BRIDGE_WARN(
        "High water mark for synapse events ",
        synapse_helpers::event_handle_cache::get_num_events_high_watermark(),
        " reached, will not create any time event");
    return nullptr;
  }
}

bool is_device_acquired() {
  return static_cast<bool>(device_context.device_);
}

void synchronize_device() {
  HABANA_ASSERT(device_context.device_);
  device_context.device_->synchronize();
}
} // namespace HPUDeviceContext
} // namespace habana
