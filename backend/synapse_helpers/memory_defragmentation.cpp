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
#include "backend/synapse_helpers/memory_defragmentation.h"
#include <optional>

namespace synapse_helpers {
namespace defragment_helpers {
static std::string MemoryStateToString(MemoryState state) {
  switch (state) {
    case MemoryState::FREE:
      return "Free";
    case MemoryState::IN_USE:
      return "In use";
    case MemoryState::FIXED:
      return "Fixed";
    default:
      HABANA_ASSERT(false);
      break;
  }

  return {};
}

MemoryBlock::MemoryBlock(const MemoryBlock& other)
    : state_{other.state_},
      handle_{other.handle_},
      ptr_{other.ptr_},
      size_{other.size_},
      actual_size_{other.actual_size_},
      stream_{other.stream_} {}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
  if (this == &other) {
    return *this;
  }

  state_ = other.state_;
  handle_ = other.handle_;
  ptr_ = other.ptr_;
  size_ = other.size_;
  actual_size_ = other.actual_size_;
  stream_ = other.stream_;

  return *this;
}

MemoryBlock::MemoryBlock(
    MemoryState state,
    synapse_helpers::mem_handle::id_t handle,
    int8_t* ptr,
    size_t size,
    size_t actual_size,
    hpuStream_t stream)
    : state_(state),
      handle_(handle),
      ptr_(ptr),
      size_(size),
      actual_size_(actual_size),
      stream_(stream) {}

std::string MemoryBlock::DebugString() const {
  std::string ret;
  ret += "Memory block state :" + MemoryStateToString(state_) + "\n";
  ret += "Start pointer: " +
      std::to_string(reinterpret_cast<std::uintptr_t>(ptr_)) + "\n";
  ret += "Requested memory block size: " + std::to_string(size_) + "\n";
  ret += "Actual memory block size: " + std::to_string(actual_size_) + "\n";
  return ret;
}

std::string Region::DebugString() const {
  std::string ret;
  ret += "Used memory: " + std::to_string(in_use_memory_) + "\n";
  ret += "Free memory: " + std::to_string(free_memory_) + "\n";
  ret += "\n";
  for (auto it = begin_;; ++it) {
    ret += it->DebugString();
    if (it == end_) {
      break;
    }
  }
  return ret;
}

MemoryDefragementer::MemoryDefragementer(
    pool_allocator::SubAllocator& allocator,
    const HandlesMap& handle2pointer,
    size_t alignment)
    : allocator_(allocator),
      handle2pointer_(handle2pointer),
      alignment_(alignment) {
  auto allocation_regions = allocator_.get_memory_info();
  if (allocation_regions.size() != 1) {
    // Expected only 1 memory region, HPU does not allow memory extension.
    PT_DEVMEM_FATAL(
        "Defragmentation cannot be started. Only 1 memory region expected, got: ",
        allocation_regions.size());
  }

  // Get region information
  void* region_ptr = nullptr;
  size_t region_size = 0;
  auto& region = allocation_regions[0];
  std::tie(region_ptr, region_size) = region;

  mem_start_ptr_ = static_cast<int8_t*>(region_ptr);
  mem_end_ptr_ = static_cast<int8_t*>(mem_start_ptr_) + region_size;
  PT_DEVMEM_DEBUG("Memory Region size::", region_size);
  PT_DEVMEM_DEBUG(
      "Memstart pointer::",
      (void*)mem_start_ptr_,
      "MemEnd pointer::",
      (void*)mem_end_ptr_);

  // Workspace is allocated at the end of memory.
  // So if there is allocation at the end of memory, it is workspace.
  void* workspace_ptr = nullptr;
  std::tie(workspace_ptr, workspace_size_) = allocator_.get_tail_chunk_info();
  workspace_ptr_ = static_cast<int8_t*>(workspace_ptr);
  PT_DEVMEM_DEBUG("MemoryDefragementer workspace ptr::", (void*)workspace_ptr_);

  // There is also reserved region for small allocations.
  // If an allocation falls into category of small allocations, this region has
  // to be defragemented separately.
  void* small_allocs_ptr = nullptr;
  std::tie(small_allocs_ptr, small_allocs_size_, small_allocs_threshold_) =
      allocator_.get_small_alloc_info();
  small_allocs_ptr_ = static_cast<int8_t*>(small_allocs_ptr);
  if (small_allocs_ptr_ &&
      (small_allocs_ptr_ < mem_start_ptr_ ||
       small_allocs_ptr_ + small_allocs_size_ > mem_end_ptr_)) {
    PT_DEVMEM_FATAL(
        "Defragmentation cannot be started. Invalid memory information.");
  }
}

bool MemoryDefragementer::CollectMemoryInformation(
    std::vector<MemoryBlock>& result) {
  void* workspace_ptr = nullptr;
  size_t size;
  std::tie(workspace_ptr, size) = allocator_.get_tail_chunk_info();
  PT_DEVMEM_DEBUG("MemoryDefragementer workspace ptr::", (void*)workspace_ptr);
  std::vector<MemoryBlock> in_use_memory_blocks;
  CollectResourceInformation(in_use_memory_blocks);
  CreateMemoryMap(in_use_memory_blocks, result);
  return ValidateMemoryMap(result);
}

bool MemoryDefragementer::CollectResourceInformation(
    std::vector<MemoryBlock>& in_use_memory_blocks) {
  for (auto const& h2p : handle2pointer_) {
    if (h2p.ptr_size_.ptr_ == nullptr) {
      // Deferred allocation case.
      // alloc() was called, but the actual allocation happens only when a
      // pointer is obtained for the first time. In such case, allocation list
      // will have handle reserved, but a pointer is still nullptr.
      continue;
    }
    auto mem_state = h2p.fixed_ ? MemoryState::FIXED : MemoryState::IN_USE;
    auto mem_ptr = static_cast<int8_t*>(h2p.ptr_size_.ptr_);
    auto mem_size = h2p.ptr_size_.size_;
    PT_DEVMEM_DEBUG(
        "Collect memory info:: ptr::",
        (void*)mem_ptr,
        " WS ptr::",
        (void*)workspace_ptr_);
    // dont add it to this block if it is a workspace
    if (mem_ptr != workspace_ptr_) {
      auto mem_actual_size = allocator_.allocated_size(h2p.ptr_size_.ptr_);
      if (mem_ptr < small_allocs_ptr_ ||
          mem_ptr >= small_allocs_ptr_ + small_allocs_size_) {
        mem_actual_size = allocator_.allocated_size(h2p.ptr_size_.ptr_);
      }

      // dont add it to this block if it is a workspace
      PT_DEVMEM_DEBUG(
          "Collect memory info:: ptr:: ",
          (void*)mem_ptr,
          " Mem Size:: ",
          mem_size,
          " State:: ",
          MemoryStateToString(mem_state),
          " Actual size:: ",
          mem_actual_size);
      in_use_memory_blocks.emplace_back(
          mem_state,
          h2p.id_,
          mem_ptr,
          mem_size,
          mem_actual_size,
          h2p.ptr_size_.stream_);
    }
  }

  if (workspace_size_ > 0) {
    // Calculating workspace information.
    // Workspace is placed at the end of memory.
    PT_DEVMEM_DEBUG(
        "Collect memory info:: ptr:: ",
        (void*)workspace_ptr_,
        " Mem Size:: ",
        workspace_size_,
        " State:: ",
        MemoryStateToString(MemoryState::FIXED),
        " Actual size:: ",
        workspace_size_);
    in_use_memory_blocks.emplace_back(
        MemoryState::FIXED,
        0,
        workspace_ptr_,
        workspace_size_,
        workspace_size_,
        0);
  }

  std::sort(in_use_memory_blocks.begin(), in_use_memory_blocks.end());

  return true;
}

bool MemoryDefragementer::CreateMemoryMap(
    std::vector<MemoryBlock>& in_use_memory_blocks,
    std::vector<MemoryBlock>& memory_blocks) {
  // helpers
  auto ptr_diff = [](int8_t* ptr1, int8_t* ptr2) -> size_t {
    return reinterpret_cast<std::uintptr_t>(ptr1) -
        reinterpret_cast<std::uintptr_t>(ptr2);
  };

  auto ptr_add_offset = [](int8_t* ptr, size_t offset) -> int8_t* {
    return static_cast<int8_t*>(ptr) + offset;
  };

  // collect information about free memory blocks
  memory_blocks.clear();

  if (in_use_memory_blocks.empty()) {
    auto mem_block_size = ptr_diff(mem_end_ptr_, mem_start_ptr_);
    memory_blocks.emplace_back(
        MemoryState::FREE,
        0,
        mem_start_ptr_,
        mem_block_size,
        mem_block_size,
        0);
    PT_DEVMEM_DEBUG(
        "CreateMemoryMap:: ptr:: ",
        (void*)mem_start_ptr_,
        " Mem Size:: ",
        mem_block_size,
        " State:: ",
        MemoryStateToString(MemoryState::FREE),
        " Actual sie::",
        mem_block_size);

    return true;
  }

  // handling a case of free memory block before first memory block in use
  auto& first_memory_block = in_use_memory_blocks.front();
  if (first_memory_block.ptr_ != mem_start_ptr_) {
    auto mem_block_size = ptr_diff(first_memory_block.ptr_, mem_start_ptr_);
    memory_blocks.emplace_back(
        MemoryState::FREE,
        0,
        mem_start_ptr_,
        mem_block_size,
        mem_block_size,
        0);
    PT_DEVMEM_DEBUG(
        "CreateMemoryMap:: ptr:: ",
        (void*)mem_start_ptr_,
        " Mem Size:: ",
        mem_block_size,
        " State:: ",
        MemoryStateToString(MemoryState::FREE),
        " Actual size:: ",
        mem_block_size);
  }

  for (auto it = in_use_memory_blocks.begin(); it != in_use_memory_blocks.end();
       ++it) {
    // checking if previous memory block was free
    if (it != in_use_memory_blocks.begin()) {
      auto prev_it = it - 1;
      auto ptr_next = ptr_add_offset(prev_it->ptr_, prev_it->actual_size_);
      auto small_alloc_block = small_allocs_ptr_ + small_allocs_size_;
      while (it->ptr_ > ptr_next) {
        auto mem_block_size = ptr_diff(it->ptr_, ptr_next);
        if (ptr_next < small_alloc_block) {
          if (it->ptr_ <= small_alloc_block) {
            mem_block_size = ptr_diff(it->ptr_, ptr_next);
          } else {
            mem_block_size = ptr_diff(small_alloc_block, ptr_next);
          }
        }
        if (mem_block_size == 0)
          break;
        memory_blocks.emplace_back(
            MemoryState::FREE, 0, ptr_next, mem_block_size, mem_block_size, 0);
        PT_DEVMEM_DEBUG(
            "CreateMemoryMap:: ptr:: ",
            (void*)ptr_next,
            " Mem Size:: ",
            mem_block_size,
            " State::",
            MemoryStateToString(MemoryState::FREE),
            " Actual size:: ",
            mem_block_size);
        ptr_next = ptr_add_offset(ptr_next, mem_block_size);
      }
    }

    // adding occupied memory block
    memory_blocks.emplace_back(
        it->state_,
        it->handle_,
        it->ptr_,
        it->size_,
        it->actual_size_,
        it->stream_);
    PT_DEVMEM_DEBUG(
        "CreateMemoryMap:: ptr:: ",
        (void*)it->ptr_,
        " Mem Size:: ",
        it->size_,
        " State::",
        MemoryStateToString(it->state_),
        " Actual size:: ",
        it->actual_size_,
        " Stream",
        it->stream_);
  }

  // handling a case of free memory block after last occupied memory block
  auto& last_memory_block = in_use_memory_blocks.back();
  auto last_alloc_ptr_end =
      ptr_add_offset(last_memory_block.ptr_, last_memory_block.actual_size_);
  if (last_alloc_ptr_end < mem_end_ptr_) {
    auto mem_block_size = ptr_diff(mem_end_ptr_, last_alloc_ptr_end);
    memory_blocks.emplace_back(
        MemoryState::FREE,
        0,
        last_alloc_ptr_end,
        mem_block_size,
        mem_block_size,
        0);
    PT_DEVMEM_DEBUG(
        "CreateMemoryMap:: ptr:: ",
        (void*)last_alloc_ptr_end,
        " Mem Size::",
        mem_block_size,
        " State::",
        MemoryStateToString(MemoryState::FREE),
        " Actual size::",
        mem_block_size);
  }

  return true;
}

bool MemoryDefragementer::ValidateMemoryMap(
    std::vector<MemoryBlock>& memory_blocks) {
  if (memory_blocks.empty()) {
    return true;
  }

  auto ptr = static_cast<int8_t*>(memory_blocks.front().ptr_);
  auto size = memory_blocks.front().actual_size_;

  if (ptr != mem_start_ptr_) {
    std::string err(
        "Invalid memory information. Expected pointer: " +
        std::to_string(reinterpret_cast<std::uintptr_t>(mem_start_ptr_)) +
        ", got: " + std::to_string(reinterpret_cast<std::uintptr_t>(ptr)));
    PT_DEVMEM_DEBUG(err);
    return false;
  }

  for (auto it = memory_blocks.begin() + 1; it != memory_blocks.end(); ++it) {
    if (it->ptr_ != ptr + size) {
      std::string err(
          "Invalid memory information. Expected pointer: " +
          std::to_string(reinterpret_cast<std::uintptr_t>(ptr + size)) +
          ", got: " +
          std::to_string(reinterpret_cast<std::uintptr_t>(it->ptr_)));
      PT_DEVMEM_DEBUG(err);
      return false;
    }

    ptr = static_cast<int8_t*>(it->ptr_);
    size = it->actual_size_;
  }

  if (ptr + size != mem_end_ptr_) {
    std::string err(
        "Invalid memory information. Expected pointer: " +
        std::to_string(reinterpret_cast<std::uintptr_t>(mem_end_ptr_ - size)) +
        ", got: " + std::to_string(reinterpret_cast<std::uintptr_t>(ptr)));
    PT_DEVMEM_DEBUG(err);
    return false;
  }

  return true;
}

bool MemoryDefragementer::SelectRegionForResourceAllocation(
    std::vector<MemoryBlock>& memory_blocks,
    size_t allocation_size,
    int8_t* ptr_start,
    int8_t* ptr_end,
    bool& defragmentation_needed,
    std::unique_ptr<Region>& result) {
  auto reset_region = [](std::vector<MemoryBlock>& memory_blocks, Region& r) {
    r.begin_ = memory_blocks.end();
    r.end_ = memory_blocks.begin();
    r.in_use_memory_ = 0;
    r.free_memory_ = 0;
  };

  std::optional<Region> region;

  Region r;
  reset_region(memory_blocks, r);
  PT_DEVMEM_DEBUG(
      "START SelectRegionForResourceAllocation mem start ptr:: ",
      (void*)ptr_start,
      " mem end ptr:: ",
      (void*)ptr_end);
  for (auto it = memory_blocks.begin(); it != memory_blocks.end();) {
    auto& mem_info = *it;
    if (mem_info.ptr_ < ptr_start) {
      ++it;
      continue;
    }

    if (mem_info.ptr_ >= ptr_end) {
      break;
    }

    // look for the first free memory block
    if (r.begin_ == memory_blocks.end()) {
      if (mem_info.state_ != MemoryState::FREE) {
        ++it;
        continue;
      }

      r.begin_ = it;
    }

    if (mem_info.state_ == MemoryState::FIXED) {
      // reset statistics if fixed memory region was found
      reset_region(memory_blocks, r);
      ++it;
      continue;
    }

    // calculate region statistics
    if (mem_info.state_ == MemoryState::FREE) {
      r.free_memory_ += mem_info.actual_size_;
      PT_DEVMEM_DEBUG(
          "free memory:: ptr:: ",
          (void*)mem_info.ptr_,
          " Mem size:: ",
          mem_info.actual_size_,
          " FreeSize:: ",
          r.free_memory_);
    } else {
      r.in_use_memory_ += mem_info.actual_size_;
      PT_DEVMEM_DEBUG(
          "Used memory:: ptr:: ",
          (void*)mem_info.ptr_,
          " Mem size:: ",
          mem_info.actual_size_,
          " Used Size:: ",
          r.in_use_memory_);
    }
    ++it;
    r.end_ = it;

    // add region to the list when there is enough free memory in the region
    // to satisfy memory allocation request
    if (r.free_memory_ >= allocation_size) {
      if (!region.has_value() || r < region.value())
        region = r;

      // set iterator to the next memory block and restart looking for the next
      // memory region meeting criteria
      it = r.begin_ + 1;

      // reset region statistics
      reset_region(memory_blocks, r);
      continue;
    }
  }

  PT_DEVMEM_DEBUG("END SelectRegionForResourceAllocation Region size");
  if (!region.has_value()) {
    PT_DEVMEM_DEBUG(
        "Not enough free memory for resource allocation. Requested allocation size: ",
        allocation_size,
        " free memory: ",
        r.free_memory_);
    PT_DEVMEM_DEBUG("No region that can be defragmented was found");
    return false;
  }

  // check if defragmentation needs to be performed
  defragmentation_needed = false;
  for (auto it = region->begin_; it != region->end_; ++it) {
    if (it->state_ == defragment_helpers::MemoryState::FIXED) {
      PT_DEVMEM_FATAL(
          "Defragmentation algorithm error. Trying to move fixed memory region");
    }

    if (it->state_ == defragment_helpers::MemoryState::IN_USE) {
      defragmentation_needed = true;
    }
  }

  if (defragmentation_needed) {
    result = absl::make_unique<Region>(*region);
  }
  return true;
}

bool MemoryDefragementer::SelectRegionForWorkspaceGrow(
    std::vector<MemoryBlock>& memory_blocks,
    size_t allocation_size,
    int8_t* ptr_start,
    int8_t* ptr_end,
    bool& defragmentation_needed,
    std::unique_ptr<Region>& result) {
  size_t free_memory = 0;
  for (auto it = memory_blocks.begin(); it != memory_blocks.end(); ++it) {
    auto& mem_info = *it;
    if (mem_info.ptr_ < ptr_start) {
      continue;
    }

    if (mem_info.ptr_ >= ptr_end) {
      break;
    }

    if (mem_info.state_ == MemoryState::FIXED &&
        it + 1 == memory_blocks.end()) {
      // Skipping workspace
      break;
    }

    if (mem_info.state_ == MemoryState::FIXED) {
      // Fixed allocation resets statistics
      free_memory = 0;
      continue;
    }

    if (mem_info.state_ != MemoryState::FREE) {
      continue;
    }

    free_memory += mem_info.actual_size_;
  }

  if (not memory_blocks.empty()) {
    auto& last_mem_block = memory_blocks.back();
    if (last_mem_block.state_ == MemoryState::FIXED &&
        last_mem_block.size_ >= allocation_size) {
      defragmentation_needed = false;
      return true;
    }
  }

  auto requested_mem_extension = allocation_size - workspace_size_;
  if (free_memory < requested_mem_extension) {
    PT_DEVMEM_DEBUG(
        "Not enough free memory for workspace extension. Requested workspace extension size: ",
        requested_mem_extension,
        ", free memory: ",
        free_memory);
    return false;
  }

  // minimize region to move smallest amount of in use memory
  auto region = absl::make_unique<Region>();
  region->begin_ = memory_blocks.begin();
  region->end_ = memory_blocks.end();
  for (size_t i = memory_blocks.size(); i > 0; --i) {
    size_t idx = i - 1;
    auto& mem_info = memory_blocks[idx];
    if (mem_info.state_ == MemoryState::FIXED && i == memory_blocks.size()) {
      region->end_ = memory_blocks.begin() + idx;
      // Skipping workspace
      continue;
    }

    if (mem_info.state_ == MemoryState::FIXED) {
      PT_DEVMEM_FATAL(
          "Defragmentation algorithm error. Found fixed allocation.");
    }

    auto mem_block_size = mem_info.actual_size_;
    if (mem_info.state_ != MemoryState::FREE) {
      region->in_use_memory_ += mem_block_size;
      continue;
    }

    region->begin_ = memory_blocks.begin() + idx;
    region->free_memory_ += mem_block_size;

    if (region->free_memory_ >= requested_mem_extension) {
      break;
    }
  }

  // check if defragmentation needs to be performed
  defragmentation_needed = false;
  for (auto it = region->begin_; it != region->end_; ++it) {
    if (it->state_ == defragment_helpers::MemoryState::FIXED) {
      PT_DEVMEM_FATAL(
          "Defragmentation algorithm error. Trying to move fixed memory region");
    }

    if (it->state_ == defragment_helpers::MemoryState::IN_USE) {
      defragmentation_needed = true;
    }
  }

  if (defragmentation_needed) {
    result.swap(region);
  }

  return true;
}

/// @brief This V2 algorithm will select movable blocks before worksapce block,
/// regardless of the movable blocks are free or in-use. Then we can move those
/// selected in-use block to other free blocks, as long as have enough free
/// blocks to do the swap.
///
/// @example For exampl, this is the original memory layout: [Free 10G][Fixed
/// 10G][Free 1G][InUse 1G][WS 1G] and we want to extend the WS to 2GB, but the
/// memory block right before WS is in used, so we need defragment. With the V1
/// algo, we don't have enough free block between the WS and the Fixed block, so
/// the defragment will fail. But with V2 algo, we can select the [Free 1G] and
/// [InUse 1G] blocks, and then move the [InUse 1G] block to the [Free 10G]
/// block. Below is the memory layout after defragment: [InUse 1G][Free
/// 9G][Fixed 10G][Free 2G][WS 1G].
///
/// @note In theory, this V2 algo should not cause any regression, it just give
/// us one more chance to recorver when we encounter OOM issue.
bool MemoryDefragementer::SelectRegionForWorkspaceGrowV2(
    std::vector<MemoryBlock>& memory_blocks,
    size_t allocation_size,
    bool& defragmentation_needed,
    std::unique_ptr<Region>& result) {
  if (not memory_blocks.empty()) {
    auto& last_mem_block = memory_blocks.back();
    if (last_mem_block.state_ == MemoryState::FIXED &&
        last_mem_block.size_ >= allocation_size) {
      defragmentation_needed = false;
      return true;
    }
  }

  auto requested_mem_extension = allocation_size - workspace_size_;

  // minimize region to move smallest amount of in use memory
  auto region = absl::make_unique<Region>();
  region->begin_ = memory_blocks.begin();
  region->end_ = memory_blocks.end();
  for (size_t i = memory_blocks.size(); i > 0; --i) {
    size_t idx = i - 1;
    auto& mem_info = memory_blocks[idx];
    if (mem_info.state_ == MemoryState::FIXED && i == memory_blocks.size()) {
      region->end_ = memory_blocks.begin() + idx;
      // Skipping workspace
      continue;
    }

    if (mem_info.state_ == MemoryState::FIXED) {
      PT_DEVMEM_DEBUG(
          "Not enough movable memory for workspace extension. Requested workspace extension size: ",
          requested_mem_extension,
          ". Movable memory size: ",
          region->in_use_memory_ + region->free_memory_);
      return false;
    }

    auto mem_block_size = mem_info.actual_size_;
    if (mem_info.state_ != MemoryState::FREE) {
      region->in_use_memory_ += mem_block_size;
    } else {
      region->free_memory_ += mem_block_size;
    }

    region->begin_ = memory_blocks.begin() + idx;

    if (region->in_use_memory_ + region->free_memory_ >=
        requested_mem_extension) {
      break;
    }
  }

  // check if defragmentation needs to be performed
  defragmentation_needed = false;
  for (auto it = region->begin_; it != region->end_; ++it) {
    if (it->state_ == defragment_helpers::MemoryState::FIXED) {
      PT_DEVMEM_FATAL(
          "Defragmentation algorithm error. Trying to move fixed memory region");
    }

    if (it->state_ == defragment_helpers::MemoryState::IN_USE) {
      defragmentation_needed = true;
    }
  }

  if (defragmentation_needed) {
    result.swap(region);
  }

  return true;
}

bool MemoryDefragementer::Run(
    std::vector<MemoryBlock>& memory_blocks,
    bool workspace_grow,
    size_t allocation_size,
    bool& defragmentation_needed,
    std::unique_ptr<Region>& result,
    bool& is_v2) {
  // update the allocation size with alignment
  allocation_size = ((allocation_size + alignment_ - 1) & ~(alignment_ - 1));
  if (workspace_grow) {
    bool found = SelectRegionForWorkspaceGrow(
        memory_blocks,
        allocation_size,
        small_allocs_ptr_ + small_allocs_size_,
        mem_end_ptr_,
        defragmentation_needed,
        result);
    if (found) {
      is_v2 = false;
      return found;
    }
    is_v2 = true;
    return SelectRegionForWorkspaceGrowV2(
        memory_blocks,
        allocation_size,
        defragmentation_needed,
        result);
  }

  if (small_allocs_ptr_ && allocation_size <= small_allocs_threshold_) {
    PT_DEVMEM_DEBUG("Checking if small allocations region can be defragmented");
    if (SelectRegionForResourceAllocation(
            memory_blocks,
            allocation_size,
            small_allocs_ptr_,
            small_allocs_ptr_ + small_allocs_size_,
            defragmentation_needed,
            result)) {
      return true;
    }

    PT_DEVMEM_WARN("No space in small allocations region.");
  }

  if (small_allocs_ptr_ == mem_start_ptr_) {
    return SelectRegionForResourceAllocation(
        memory_blocks,
        allocation_size,
        small_allocs_ptr_ + small_allocs_size_,
        mem_end_ptr_,
        defragmentation_needed,
        result);
  } else {
    // Not expected case, but it needs to be handled for completeness
    return SelectRegionForResourceAllocation(
        memory_blocks,
        allocation_size,
        mem_start_ptr_,
        small_allocs_ptr_,
        defragmentation_needed,
        result);
  }

  return true;
}

} // namespace defragment_helpers
} // namespace synapse_helpers
