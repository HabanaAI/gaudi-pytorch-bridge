/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
