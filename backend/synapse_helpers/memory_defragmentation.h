/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "backend/synapse_helpers/mem_handle.h"
#include "habana_helpers/logging.h"
#include "pool_allocator/PoolAllocator.h"

namespace synapse_helpers {
namespace defragment_helpers {

enum class MemoryState { FREE, IN_USE, FIXED };

struct MemoryBlock {
  MemoryBlock() = default;
  ~MemoryBlock() = default;
  MemoryBlock(const MemoryBlock& other);
  MemoryBlock& operator=(const MemoryBlock& other);

  MemoryBlock(
      MemoryState state,
      synapse_helpers::mem_handle::id_t handle,
      int8_t* ptr,
      size_t size,
      size_t actual_size,
      hpuStream_t stream);

  std::string DebugString() const;

  bool operator<(const MemoryBlock& rhs) const {
    return ptr_ < rhs.ptr_;
  }

  MemoryState state_ = MemoryState::FREE;
  synapse_helpers::mem_handle::id_t handle_ = 0;
  int8_t* ptr_ = nullptr;
  size_t size_ = 0;
  size_t actual_size_ = 0;
  hpuStream_t stream_ = 0;
};

struct Region {
  std::vector<MemoryBlock>::iterator begin_;
  std::vector<MemoryBlock>::iterator end_;
  size_t in_use_memory_ = 0;
  size_t free_memory_ = 0;

  bool operator<(const Region& rhs) const {
    return in_use_memory_ < rhs.in_use_memory_;
  }

  std::string DebugString() const;
};

class MemoryDefragementer {
 public:
  MemoryDefragementer() = delete;
  MemoryDefragementer(
      pool_allocator::SubAllocator& allocator,
      const HandlesMap& handle2pointer,
      size_t alignment);

  bool CollectMemoryInformation(std::vector<MemoryBlock>& result);
  bool Run(
      std::vector<MemoryBlock>& memory_blocks,
      bool workspace_grow,
      size_t allocation_size,
      bool& defragmentation_needed,
      std::unique_ptr<Region>& result,
      bool& is_v2);

 private:
  pool_allocator::SubAllocator& allocator_;
  const HandlesMap& handle2pointer_;
  size_t alignment_;

  int8_t* mem_start_ptr_ = nullptr;
  int8_t* mem_end_ptr_ = nullptr;

  int8_t* workspace_ptr_ = nullptr;
  size_t workspace_size_ = 0;

  int8_t* small_allocs_ptr_ = nullptr;
  size_t small_allocs_size_ = 0;
  size_t small_allocs_threshold_ = 0;

  bool CollectResourceInformation(
      std::vector<MemoryBlock>& in_use_memory_blocks);
  bool CreateMemoryMap(
      std::vector<MemoryBlock>& in_use_memory_blocks,
      std::vector<MemoryBlock>& result);
  bool ValidateMemoryMap(std::vector<MemoryBlock>& memory_blocks);

  bool SelectRegionForResourceAllocation(
      std::vector<MemoryBlock>& memory_blocks,
      size_t allocation_size,
      int8_t* ptr_start,
      int8_t* ptr_end,
      bool& defragmentation_needed,
      std::unique_ptr<Region>& result);
  bool SelectRegionForWorkspaceGrow(
      std::vector<MemoryBlock>& memory_blocks,
      size_t allocation_size,
      int8_t* ptr_start,
      int8_t* ptr_end,
      bool& defragmentation_needed,
      std::unique_ptr<Region>& result);
  bool SelectRegionForWorkspaceGrowV2(
      std::vector<MemoryBlock>& memory_blocks,
      size_t allocation_size,
      bool& defragmentation_needed,
      std::unique_ptr<Region>& result);
};

} // namespace defragment_helpers
} // namespace synapse_helpers
