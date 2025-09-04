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
#pragma once

#include <synapse_common_types.h>

#include <common/strong_type.h>
#include <mutex>
#include <set>
#include <unordered_map>

namespace synapse_helpers {
class device_interface;

class host_memory {
 public:
  explicit host_memory(device_interface& device);
  ~host_memory();
  host_memory(const host_memory&) = delete;
  host_memory& operator=(const host_memory&) = delete;
  host_memory(host_memory&&) = delete;
  host_memory& operator=(host_memory&&) = delete;
  synStatus malloc(void** ptr, size_t size);
  synStatus free(void* ptr);
  synStatus uncached_malloc(void** ptr, size_t size);
  synStatus uncached_free(void* ptr);
  void dropCache();
  bool is_host_memory(void* ptr);

 private:
  struct BlockSize {
    size_t size; // allocation size
    void* ptr; // host memory pointer

    BlockSize(size_t size, void* ptr = nullptr) : size(size), ptr(ptr) {}
  };

  struct Block : public BlockSize {
    using is_allocated_t = common::StrongType<bool, struct IsAllocatedTTag>;
    using is_huge_page_t = common::StrongType<bool, struct IsHugePageTTag>;
    using is_real_allocation_t =
        common::StrongType<bool, struct IsRealAllocationTTag>;
    is_allocated_t allocated; // true if the block is currently allocated
    is_huge_page_t is_huge_page;
    is_real_allocation_t
        real_allocation; // true if the block is a real allocation, false is
                         // part of allocated buffer
    size_t real_allocation_size;
    void* real_allocation_ptr;
    size_t ref_count{0};

    Block(
        size_t size,
        void* ptr,
        is_allocated_t allocated,
        is_huge_page_t is_huge_page,
        is_real_allocation_t real_allocation,
        size_t real_allocation_size,
        void* real_allocation_ptr)
        : BlockSize(size, ptr),
          allocated(allocated),
          is_huge_page(is_huge_page),
          real_allocation(real_allocation),
          real_allocation_size(real_allocation_size),
          real_allocation_ptr(real_allocation_ptr) {}
  };

  // Allocates memory on host and maps it to Synapse.
  // For allocations larger than 2MB tries to allocate huge page. If this
  // succeed, updates actual_allocation_size to aligned size of allocation. If
  // huge page allocation is unnecessary or not possible (e.g, error, or
  // exhausted limit), resolves to synHostMalloc.
  std::tuple<synStatus, Block::is_huge_page_t> alloc_memory(
      size_t& actual_allocation_size,
      void** ptr);

  void free_memory(
      void* const ptr,
      Block::is_huge_page_t is_huge_page,
      size_t size);

  static bool BlockComparator(const BlockSize& a, const BlockSize& b) {
    // sort by size, break ties with pointer
    if (a.size != b.size) {
      return a.size < b.size;
    }
    return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
  }
  using Comparison = bool (*)(const BlockSize&, const BlockSize&);

  // lock around all operations
  std::mutex mutex_;

  device_interface& device_;

  // Pointers that are ready to be allocated
  std::set<BlockSize, Comparison> available_;

  // Blocks by pointer
  std::unordered_map<void*, Block> blocks_;

  // In case of huge pages, we try to split it equally between workers.
  // This ignores other huge pages users in the system, but should give quite
  // good approximation so each worker won't starve. Optionally can be overriden
  // by env var PT_HPU_HUGE_PAGES_LIMIT_MB.
  const size_t available_huge_pages_mb_for_worker_;

  // Remaining huge pages for use by this worker. Initially set to
  // available_huge_pages_mb_for_worker_.
  size_t remaining_huge_pages_mb_;
};
} // namespace synapse_helpers
