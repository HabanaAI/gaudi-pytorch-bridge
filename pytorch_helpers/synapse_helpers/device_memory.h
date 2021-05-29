/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
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

#include "pool_allocator/CoalescedPoolAllocator.h"
#include "pool_allocator/PoolAllocator.h"
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
  bool is_mem_threshold_hit();

 private:
  device& device_;
  pool_allocator::PoolStrategyType pool_strategy_;
  uint64_t pool_size_;
  pool_allocator::SubAllocator* suballoc_;
  bool enable_mem_threshold_check;
};
} // namespace synapse_helpers
