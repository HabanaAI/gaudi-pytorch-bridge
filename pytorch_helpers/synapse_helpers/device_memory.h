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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "pool_allocator/CoalescedPoolAllocator.h"
#include "pool_allocator/CoalescedStringentPoolAllocator.h"
#include "pool_allocator/PoolAllocator.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/id_generator.h"
#include "synapse_helpers/mem_handle.h"
#include "synapse_helpers/synapse_error.h"
#include "synapse_helpers/synchronous_counter.h"

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
  synStatus free(void* ptr);
  void* workspace_alloc(void* ptr, size_t& ws_size, size_t req_size);
  void fix_address(void* ptr);
  bool is_mem_threshold_hit();
  void get_memory_stats(MemoryStats* stats);
  void clear_memory_stats();
  device_ptr_lock lock_addresses(const std::vector<device_ptr>&);
  using ptr_with_size = std::pair<void*, size_t>;
  using handle2pointer_map =
      absl::flat_hash_map<mem_handle::id_t, ptr_with_size>;
  pool_allocator::PoolStrategyType get_pool_strategy() {
    return pool_strategy_;
  }
  void reset_pool();

 private:
  device& device_;
  pool_allocator::PoolStrategyType pool_strategy_;
  uint64_t pool_size_;
  pool_allocator::SubAllocator* suballoc_;
  bool enable_mem_threshold_check;

  std::mutex mutex_;
  std::mutex defragmentation_mutex_;
  id_generator<mem_handle::id_t> handle_id_generator_;
  handle2pointer_map handle2pointer_;
  device_ptr workspace_allocation_;
  device_ptr get_pointer(mem_handle);
  synStatus alloc(void** v_ptr, uint64_t size, bool is_workspace = false);
  synStatus deallocate(void* ptr);
  void check_and_limit_recipe_execution(void);
  bool defragment_memory(
      size_t alignment,
      size_t allocation_size,
      bool workspace_grow);
  std::shared_ptr<synapse_helpers::synchronous_counter>
      threads_in_defragmenter_critical_section_ =
          std::make_shared<synapse_helpers::synchronous_counter>();
  absl::flat_hash_set<mem_handle::id_t> fixed_handles_;
};
} // namespace synapse_helpers
