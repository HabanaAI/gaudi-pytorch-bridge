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
#include <synapse_common_types.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/device_mem_reporter.h"
#include "backend/synapse_helpers/mem_handle.h"
#include "backend/synapse_helpers/stream.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "backend/synapse_helpers/synchronous_counter.h"
#include "mem_hlml.h"
#include "pool_allocator/PoolAllocator.h"

namespace synapse_helpers {
class device;

class device_memory {
 public:
  explicit device_memory(device& device);
  ~device_memory(); //= default;
  device_memory(const device_memory&) = delete;
  device_memory& operator=(const device_memory&) = delete;
  device_memory(device_memory&&) = delete;
  device_memory& operator=(device_memory&&) = delete;
  synStatus malloc(void** ptr, size_t size);
  synStatus malloc(
      void** ptr,
      size_t size,
      synapse_helpers::hpuStream_t stream);
  synStatus free(void* ptr);
  synStatus free_with_stream(void* ptr);
  void* workspace_alloc(void* ptr, size_t& ws_size, size_t req_size);
  synStatus workspace_free(void* ptr);
  device_ptr fix_address(void* ptr);
  void record_param(
      const std::string& name,
      const bool is_param,
      const bool is_grad,
      const bool is_optim_state,
      const uint64_t t_start,
      const uint64_t t_end);
  void get_memory_stats(MemoryStats* stats);
  std::vector<std::pair<uint64_t, uint64_t>> get_occupied_chunk_map();
  void clear_memory_stats();
  void reset_peak_memory_stats();
  device_ptr_lock lock_addresses(absl::Span<const device_ptr>);
  pool_allocator::PoolStrategyType get_pool_strategy() {
    return pool_strategy_;
  }
  void reset_pool();
  size_t get_total_memory_required(absl::Span<const device_ptr>);
  size_t block_align(size_t n);
  bool is_memory_available(size_t size);
  bool is_memory_available(
      size_t persistent_size,
      size_t curr_ws_size,
      size_t new_ws_size);
  bool is_allocated(const device_ptr address) const;
  size_t get_max_cntgs_chunk_size() const;
  synapse_helpers::MemoryReporter* get_memory_reporter();
  void recordStream(void* ptr, synapse_helpers::hpuStream_t stream);

  std::shared_ptr<HlMlMemoryReporter> get_hlml_memory_reporter() const {
    return m_hlml_memory_reporter;
  }

 private:
  device& device_;
  pool_allocator::PoolStrategyType pool_strategy_;
  uint64_t pool_size_;
  pool_allocator::SubAllocator* suballoc_;
  bool enable_mem_threshold_check;

  std::mutex mutex_;
  std::mutex defragmentation_mutex_;
  bool update_on_defragment_ = false;
  HandlesMap handle2pointer_;
  device_ptr get_pointer(mem_handle);
  synStatus alloc(void** v_ptr, uint64_t size, bool is_workspace = false);
  synStatus alloc(void** v_ptr, uint64_t size, hpuStream_t stream);
  synStatus deallocate(void* ptr);
  synStatus deallocate(void* ptr, hpuStream_t stream);
  void check_and_limit_recipe_execution(size_t size);
  bool defragment_memory(
      size_t alignment,
      size_t allocation_size,
      bool workspace_grow);
  void MoveData(
      device& dev,
      std::vector<std::tuple<uint64_t, uint64_t, size_t>> move_address);
  std::shared_ptr<synapse_helpers::synchronous_counter>
      threads_in_defragmenter_critical_section_ =
          std::make_shared<synapse_helpers::synchronous_counter>();
  void record(void* ptr, size_t size, bool alloc);

  MemoryReporter mem_reporter;
  void init_hlml_memory();
  std::shared_ptr<HlMlMemoryReporter> m_hlml_memory_reporter;
  std::shared_ptr<HlMlMemoryUpdater> m_hlml_memory_updater;
  size_t alignment_;
};
} // namespace synapse_helpers
