/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include <c10/util/thread_name.h>
#include <synapse_api_types.h>
#include <memory>
#include "backend/habana_device/HPUAllocator.h"
#include "backend/habana_device/PinnedMemoryAllocator.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/kernel/constant_information.h"
#include "backend/scalar_cache.h"
#include "backend/synapse_helpers/time_slot.h"
#include "common/pipeline_deleter.h"

namespace habana::HPUDeviceContext {

class HPUDeviceContextImpl {
 public:
  static HPUDeviceContextImpl& instance() {
    if (!device_context) {
      device_context.reset(new HPUDeviceContextImpl);
    }
    return *device_context;
  }

  HPUDeviceContextImpl(const HPUDeviceContextImpl&) = delete;
  HPUDeviceContextImpl(HPUDeviceContextImpl&&) = delete;
  HPUDeviceContextImpl& operator=(const HPUDeviceContextImpl&) = delete;
  HPUDeviceContextImpl& operator=(HPUDeviceContextImpl&&) = delete;
  ~HPUDeviceContextImpl() = default;
  std::unique_ptr<habana_helpers::SingleThreadPool> garbage_collection_thread_;
  synapse_helpers::device_handle device_;
  std::unique_ptr<backend::ScalarCache> scalar_cache_;
  std::unique_ptr<backend::H2dScalesCache> h2d_scales_cache_;

  std::unique_ptr<RecipeCacheLRU> recipe_cache_;

  std::unique_ptr<habana_helpers::ThreadPool> lazy_compile_thread_pool_;

  std::unique_ptr<PipeSingleThreadpool> execute_thread_;
  std::unique_ptr<PipeThreadpool> compile_thread_pool_;
  std::unique_ptr<PipeSingleThreadpool> lowering_thread_;

  // Holding this is required for proper destruction order
  std::shared_ptr<ConstantInformation> constant_information_;
  bool exception_occurred_ = false;
  void Init();
  void JoinAllThreads();
  void JoinPipelineThreads();
  void JoinLoweringThread();
  void CreateDevice();
  void Finish();
  void ThreadsRelease();

 private:
  HPUDeviceContextImpl() = default;
  static std::unique_ptr<HPUDeviceContextImpl> device_context;
};

std::unique_ptr<HPUDeviceContextImpl> HPUDeviceContextImpl::device_context{};

void HPUDeviceContextImpl::JoinAllThreads() {
  if (!lowering_thread_) {
    return;
  }
  JoinPipelineThreads();
  garbage_collection_thread_->waitWorkComplete();
}
void HPUDeviceContextImpl::JoinPipelineThreads() {
  if (!lowering_thread_) {
    return;
  }
  try {
    lowering_thread_->waitWorkComplete();
  } catch (...) {
    exception_occurred_ = true;
    throw;
  }

  compile_thread_pool_->waitWorkComplete();
  execute_thread_->waitWorkComplete();
}

void HPUDeviceContextImpl::JoinLoweringThread() {
  if (!lowering_thread_) {
    return;
  }
  try {
    lowering_thread_->waitWorkComplete();
  } catch (...) {
    exception_occurred_ = true;
    throw;
  }
}

void HPUDeviceContextImpl::CreateDevice() {
  auto device_ptr_or_error = synapse_helpers::device::get_or_create(
      synapse_helpers::device::get_supported_devices());

  if (std::holds_alternative<synapse_helpers::synapse_error>(
          device_ptr_or_error)) {
    auto error = std::get<synapse_helpers::synapse_error>(device_ptr_or_error);
    TORCH_HABANA_CHECK(error.status, error.error);
  } else {
    device_ = std::get<synapse_helpers::device_handle>(device_ptr_or_error);
  }
}

void HPUDeviceContextImpl::Init() {
  CreateDevice();
  garbage_collection_thread_ =
      std::make_unique<habana_helpers::SingleThreadPool>(true);
  recipe_cache_ = std::make_unique<RecipeCacheLRU>();

  lazy_compile_thread_pool_ = std::make_unique<habana_helpers::ThreadPool>();

  execute_thread_ = std::make_unique<PipeSingleThreadpool>(true, []() {
    c10::setThreadName("Pipeline Execute Thread");
    common::PipelineDeleter::instance().install();
  });
  compile_thread_pool_ = std::make_unique<PipeThreadpool>(
      true,
      []() { c10::setThreadName("Pipeline Compile Thread"); },
      GET_ENV_FLAG_NEW(PT_HPU_COMPILE_THREAD_POOL_SIZE));
  lowering_thread_ = std::make_unique<PipeSingleThreadpool>(
      true, []() { c10::setThreadName("Pipeline Lowering Thread"); });
  constant_information_ = ConstantInformationPtr();
  scalar_cache_ = std::make_unique<backend::ScalarCache>();
  h2d_scales_cache_ = std::make_unique<backend::H2dScalesCache>();

  HPURegistrar::get_hpu_registrar().register_thread_deleter(
      []() { HPUDeviceContextImpl::instance().ThreadsRelease(); });
  HPURegistrar::get_hpu_registrar().register_device_deleter(
      []() { HPUDeviceContextImpl::instance().Finish(); });
  HPURegistrar::get_hpu_registrar().register_device_context_deleter(
      []() { device_context.reset(); });
}

void HPUDeviceContextImpl::ThreadsRelease() {
  // Make sure all pipeline tasks finished before the reset
  JoinPipelineThreads();
  common::PipelineDeleter::instance().uninstall();

  habana_helpers::AutoNoGIL gil_release;
  HPUDeviceContextImpl::instance().lowering_thread_.reset();
  HPUDeviceContextImpl::instance().compile_thread_pool_.reset();
  HPUDeviceContextImpl::instance().execute_thread_.reset();
}

void HPUDeviceContextImpl::Finish() {
  synapse_helpers::MemoryStats stats;
  device_->get_device_memory().get_memory_stats(&stats);

  // Currently statistic is device independent. device_->id() can be added if
  // needed The backend device maximum bytes in use
  const std::string pb_name("peak_bytes");
  const std::string pb_val(std::to_string(stats.peak_bytes_in_use));
  // The backend device internal memory used
  const std::string wrksp_name("workspace_bytes");
  const std::string wrksp_val(std::to_string(stats.scratch_mem_in_use));

  habana_helpers::EmitEvent(
      habana_helpers::EventDispatcher::Topic::CTX_FINISH_BEFORE,
      {{pb_name, pb_val}, {wrksp_name, wrksp_val}});

  lazy_compile_thread_pool_.reset();
  recipe_cache_.reset();
  scalar_cache_.reset();
  h2d_scales_cache_.reset();

  constant_information_->ClearChecksumInformation();

  // We have to remove garbage_collection_thread_ after destroying the stream
  // but before releasing the device_id. Both classes are owned by class device
  // So, we have to use this workaround till we refactor device class by
  // decomposing it into smaller classes
  device_->cleanup();

  garbage_collection_thread_.reset();
  if (device_.use_count() != 1) {
    TORCH_WARN(
        "when deleting HPUDevice, device is kept alive by ",
        device_.use_count() - 1,
        " other references ");
  }

  device_.reset();

  habana::HPUDeviceAllocator::allocator_active_device_id =
      SYN_INVALID_DEVICE_ID;
  habana::PinnedMemoryAllocator::allocator_active_device_id =
      SYN_INVALID_DEVICE_ID;
  constant_information_.reset();
}

synapse_helpers::device& get_device(synDeviceId /*unused*/) {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  return *HPUDeviceContextImpl::instance().device_;
}

void join_all_threads() {
  HPUDeviceContextImpl::instance().JoinAllThreads();
}
void join_pipeline_threads() {
  HPUDeviceContextImpl::instance().JoinPipelineThreads();
}

void join_lowering_thread() {
  HPUDeviceContextImpl::instance().JoinLoweringThread();
}

bool get_exception_occurred() {
  if (!is_device_acquired()) {
    return false;
  }
  bool exception_occurred =
      HPUDeviceContextImpl::instance().exception_occurred_;
  HPUDeviceContextImpl::instance().exception_occurred_ = false;
  return exception_occurred;
}

PipeThreadpool& compile_thread_pool() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().compile_thread_pool_);
  return *HPUDeviceContextImpl::instance().compile_thread_pool_;
}

habana_helpers::SingleThreadPool& garbage_collection_thread() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().garbage_collection_thread_);
  return *HPUDeviceContextImpl::instance().garbage_collection_thread_;
}

PipeSingleThreadpool& lowering_thread() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().lowering_thread_);
  return *HPUDeviceContextImpl::instance().lowering_thread_;
}

PipeSingleThreadpool& execute_thread() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().execute_thread_);
  return *HPUDeviceContextImpl::instance().execute_thread_;
}

habana_helpers::ThreadPool& lazy_compile_thread_pool() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().lazy_compile_thread_pool_);
  return *HPUDeviceContextImpl::instance().lazy_compile_thread_pool_;
}

backend::ScalarCache& scalar_cache() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().scalar_cache_);
  return *HPUDeviceContextImpl::instance().scalar_cache_;
}

backend::H2dScalesCache& h2d_scales_cache() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().h2d_scales_cache_);
  return *HPUDeviceContextImpl::instance().h2d_scales_cache_;
}

synapse_helpers::device& syn_device() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  return *HPUDeviceContextImpl::instance().device_;
}

RecipeCacheLRU& recipe_cache() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().recipe_cache_);
  return *HPUDeviceContextImpl::instance().recipe_cache_;
}

void recipe_cache_clear() {
  if (HPUDeviceContextImpl::instance().recipe_cache_) {
    HPUDeviceContextImpl::instance().recipe_cache_->clear();
  }
}

void flush_disk_cache() {
  if (HPUDeviceContextImpl::instance().recipe_cache_) {
    HPUDeviceContextImpl::instance().recipe_cache_->FlushDiskCache();
  }
}

void synchronize() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  HPUDeviceContextImpl::instance().device_->synchronize();
}

void synchronize_host_multistage_pipeline() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0) {
    HPUDeviceContextImpl::instance().JoinAllThreads();
  }
}

std::string get_device_capability() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  return HPUDeviceContextImpl::instance().device_->get_device_capability();
}

std::string get_device_properties(unsigned id) {
  return synapse_helpers::device::get_device_properties(id);
}

int get_total_device_count() {
  return synapse_helpers::device::get_total_device_count();
}

c10::Device get_or_create_aten_device() {
  PT_BRIDGE_BEGIN;
  if (!HPUDeviceContextImpl::instance().device_) {
    HPUDeviceContextImpl::instance().Init();
    PT_BRIDGE_DEBUG(
        "Created hpu device ", HPUDeviceContextImpl::instance().device_.get());
    habana::HPUDeviceAllocator::allocator_active_device_id = 0;
    habana::PinnedMemoryAllocator::allocator_active_device_id = 0;
  }
  return {at::kHPU, static_cast<at::DeviceIndex>(0)};
}

c10::Device aten_device() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
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
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  auto syn_error{HPUDeviceContextImpl::instance().device_->copy_data_to_device(
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
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  auto syn_error{HPUDeviceContextImpl::instance().device_->copy_data_to_device(
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
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  auto syn_error{HPUDeviceContextImpl::instance().device_->copy_data_to_host(
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
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  auto syn_error{
      HPUDeviceContextImpl::instance().device_->copy_data_within_device(
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
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  return HPUDeviceContextImpl::instance().device_->get_device_memory();
}

synapse_helpers::host_memory& get_host_memory() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  return HPUDeviceContextImpl::instance().device_->get_host_memory();
}

std::shared_ptr<synapse_helpers::TimeSlot> create_time_slot(
    synapse_helpers::hpuStream_t& hpu_stream) {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  auto& device = *HPUDeviceContextImpl::instance().device_;
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
  return static_cast<bool>(HPUDeviceContextImpl::instance().device_);
}

void synchronize_device() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  HPUDeviceContextImpl::instance().device_->synchronize();
}

void set_scale_attributes(bool is_hw_aligned, uint32_t scale_hash_id) {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  HPUDeviceContextImpl::instance().device_->set_scale_attributes(
      is_hw_aligned, scale_hash_id);
}

uint32_t get_scale_attribute_hash_id() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  return HPUDeviceContextImpl::instance()
      .device_->get_scale_attribute_hash_id();
}

void set_is_dynamic_quantization(bool is_dynamic_quantization) {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  HPUDeviceContextImpl::instance().device_->set_is_dynamic_quantization(
      is_dynamic_quantization);
}

bool get_is_dynamic_quantization() {
  HABANA_ASSERT(HPUDeviceContextImpl::instance().device_);
  return HPUDeviceContextImpl::instance()
      .device_->get_is_dynamic_quantization();
}
} // namespace habana::HPUDeviceContext
