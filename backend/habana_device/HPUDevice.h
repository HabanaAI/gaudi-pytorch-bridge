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
#include "backend/kernel/constant_information.h"
#include "backend/kernel/hpu_recipe_cache.h"
#include "backend/scalar_cache.h"
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"

#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"

namespace synapse_helpers {
class TimeSlot;
}

namespace habana {

class ThreadPoolWithGILRelease : public habana_helpers::ThreadPool {
 public:
  ThreadPoolWithGILRelease() : habana_helpers::ThreadPool(true){};
  void waitWorkComplete();
};

namespace HPUDeviceContext {
habana_helpers::ThreadPool& garbage_collection_thread();
ThreadPoolWithGILRelease& compile_thread();
ThreadPoolWithGILRelease& lowering_thread();
ThreadPoolWithGILRelease& execute_thread();
RecipeCacheLRU& recipe_cache();
void recipe_cache_clear();
void flush_disk_cache();
backend::ScalarCache& scalar_cache();
// TODO id should be removed
synapse_helpers::device& get_device(int id = 0);

void synchronize();
void synchronize_host_multistage_pipeline();

std::string get_device_capability();
std::string get_device_properties(unsigned id);
int get_total_device_count();

void join_all_threads();
void join_pipeline_threads();

c10::Device get_or_create_aten_device();
c10::Device aten_device();

void copy_data_to_device(
    void* cpu_data,
    synapse_helpers::device_ptr destination,
    synapse_helpers::device_ptr event_addr,
    size_t total_bytes,
    const synapse_helpers::event_done_callback& done_cb,
    bool non_blocking = false,
    bool is_pinned = false,
    synapse_helpers::hpuStream_t hpu_stream = 0,
    void* host_cpu_data = nullptr);

void copy_data_to_device(
    synapse_helpers::device::transfer_manifest const& transfers,
    synapse_helpers::event_done_callback unref_cb,
    synapse_helpers::hpuStream_t hpu_stream = 0);

void copy_data_to_host(
    synapse_helpers::device_ptr device_data,
    void* destination,
    synapse_helpers::device_ptr event_addr,
    size_t total_bytes,
    const synapse_helpers::event_done_callback& done_cb,
    bool is_pinned = false,
    synapse_helpers::hpuStream_t hpu_stream = 0);

void copy_data_within_device(
    synapse_helpers::device_ptr source,
    synapse_helpers::device_ptr destination,
    synapse_helpers::device_ptr src_event_addr,
    synapse_helpers::device_ptr dst_event_addr,
    size_t total_bytes,
    synapse_helpers::event_done_callback unref_cb,
    synapse_helpers::hpuStream_t hpu_stream = 0);

synapse_helpers::device_memory& get_device_memory();
synapse_helpers::host_memory& get_host_memory();

std::shared_ptr<synapse_helpers::TimeSlot> create_time_slot(
    synapse_helpers::hpuStream_t& hpu_stream);

bool is_device_acquired();

bool get_exception_occurred();

void synchronize_device();
}; // namespace HPUDeviceContext

} // namespace habana
