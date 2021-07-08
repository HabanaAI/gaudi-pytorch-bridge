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
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "pool_allocator/CoalescedPoolAllocator.h"
#include "pool_allocator/CoalescedStringentPoolAllocator.h"
#include "pool_allocator/PoolAllocator.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/id_generator.h"
#include "synapse_helpers/mem_handle.h"
#include "synapse_helpers/synapse_error.h"

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
  device_ptr_lock lock_addresses(const std::vector<device_ptr>&);
  using ptr_with_size = std::pair<void*, size_t>;
  using handle2pointer_map =
      absl::flat_hash_map<mem_handle::id_t, ptr_with_size>;

 private:
  device& device_;
  pool_allocator::PoolStrategyType pool_strategy_;
  uint64_t pool_size_;
  pool_allocator::SubAllocator* suballoc_;
  bool enable_mem_threshold_check;

  std::mutex mutex_;
  id_generator<mem_handle::id_t> handle_id_generator_;
  handle2pointer_map handle2pointer_;
  device_ptr workspace_allocation_;
  device_ptr get_pointer(mem_handle);
  synStatus alloc(void** v_ptr, uint64_t size);
  synStatus deallocate(void* ptr);
};
} // namespace synapse_helpers
