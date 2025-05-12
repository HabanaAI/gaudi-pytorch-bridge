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
#include "backend/synapse_helpers/host_memory.h"
#include <backend/synapse_helpers/env_flags.h>
#include <synapse_api.h>
#include <synapse_common_types.h>
#include <sys/mman.h>
#include <utility>
#include "backend/synapse_helpers/device_interface.h"
#include "backend/synapse_helpers/env_flags_impl.h"
#include "habana_helpers/logging.h"

namespace synapse_helpers {

namespace {
constexpr size_t size_1mb = 1ull * 1024 * 1024;
constexpr size_t size_2mb = 2 * size_1mb;

size_t read_nr_hugepages_file(const std::string_view file_path) {
  auto nr_hugepages_file_path = std::filesystem::path{file_path};
  if (!std::filesystem::exists(nr_hugepages_file_path)) {
    PT_SYNHELPER_WARN("{} file not found.", file_path);
    return 0ull;
  }

  std::ifstream nr_hugepages_file(nr_hugepages_file_path);
  if (!nr_hugepages_file.is_open()) {
    PT_SYNHELPER_WARN("Failed to open {} file.", file_path);
    return 0ull;
  }

  std::string line;
  std::getline(nr_hugepages_file, line);
  return std::stoull(line);
}

// Computes the limit for huge pages for each device in the system.
// Uses either env variable, is user supplied custom value or obtain HugePages
// size from the system and splits it equally between all Gaudis available in
// the system, so each device shouldn't starve. This ignores other HugePages
// users in the system (in particular GC and HCL), but should be good enough
// approximation for most cases.
size_t compute_huge_pages_limit() {
  auto manual_limit = GET_ENV_FLAG_NEW(PT_HPU_HUGE_PAGES_LIMIT_MB);
  if (manual_limit > 0) {
    PT_SYNHELPER_DEBUG(
        "Using manual limit for huge pages: {} MB", manual_limit);
    return manual_limit * 1024ull * 1024ull;
  }

  using namespace std::literals;

  const auto huge_pages_count = read_nr_hugepages_file(
      "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"sv);

  const auto surplus_huge_pages_count = read_nr_hugepages_file(
      "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages"sv);

  uint32_t device_count = 0;
  auto status = synDeviceGetCount(&device_count);

  if (synSuccess == status) {
    auto limit = ((huge_pages_count + surplus_huge_pages_count) * size_2mb) /
        device_count;
    PT_SYNHELPER_DEBUG("Huge pages limit calculated: {} MB per device", limit);
    return limit;
  } else {
    PT_SYNHELPER_WARN(
        "Failed to get device count for huge pages limit calculation. {}",
        status);
    return 0;
  }
}
} // namespace

host_memory::host_memory(device_interface& device)
    : mutex_{},
      device_{device},
      available_(BlockComparator),
      available_huge_pages_mb_for_worker_{compute_huge_pages_limit()},
      remaining_huge_pages_mb_(available_huge_pages_mb_for_worker_) {}

host_memory::~host_memory() {
  std::lock_guard<std::mutex> lock(mutex_);
  dropCache();
}

std::tuple<synStatus, host_memory::Block::is_huge_page_t> host_memory::
    alloc_memory(size_t& actual_allocation_size, void** ptr) {
  if (actual_allocation_size >= size_2mb) {
    auto aligned_actual_allocation_size =
        (actual_allocation_size + size_2mb - 1) -
        actual_allocation_size % size_2mb;
    if (remaining_huge_pages_mb_ >= aligned_actual_allocation_size) {
      constexpr auto prot = PROT_READ | PROT_WRITE;
      static const auto flags = MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB |
          (GET_ENV_FLAG_NEW(PT_HPU_HUGE_PAGES_POPULATE) ? MAP_POPULATE : 0);

      *ptr = mmap(nullptr, aligned_actual_allocation_size, prot, flags, -1, 0);

      auto errno_mmap = errno;

      if (*ptr != MAP_FAILED) {
        auto map_result =
            synHostMap(device_.id(), aligned_actual_allocation_size, *ptr);
        if (synSuccess == map_result) {
          actual_allocation_size = aligned_actual_allocation_size;
          remaining_huge_pages_mb_ -= actual_allocation_size;
          return std::make_tuple(map_result, Block::is_huge_page_t{true});
        } else {
          PT_SYNHELPER_WARN("Mapping huge page to Synapse failed.", map_result);
          munmap(*ptr, aligned_actual_allocation_size);
          *ptr = nullptr;
        }
      } else {
        *ptr = nullptr;
        PT_SYNHELPER_WARN(
            "Huge page allocation failed. ", *ptr, " ", errno_mmap);
      }
    }
  }

  auto result = synHostMalloc(device_.id(), actual_allocation_size, 0, ptr);
  if (result != synSuccess) {
    PT_SYNHELPER_WARN("SynHostMalloc Failed. ", *ptr, " ", result);
  }

  return std::make_tuple(result, Block::is_huge_page_t{false});
}

synStatus host_memory::malloc(void** ptr, const size_t size) {
  *ptr = nullptr;
  if (size == 0) {
    return synSuccess;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  size_t actual_allocation_size{size};
  bool small_alloc_optimization_enable{false};

  if (device_.HostMemoryCacheEnabled()) {
    // Search for the smallest block which can hold this allocation
    // TODO: Maybe some heuristic to split large blocks would be beneficial. Or
    // allocate if available block is too large.
    BlockSize search_key(size);
    auto it = available_.lower_bound(search_key);
    if (it != available_.end()) {
      Block& block = blocks_.at(it->ptr);
      block.allocated = Block::is_allocated_t{true};
      *ptr = block.ptr;
      available_.erase(it);
      ++blocks_.at(block.real_allocation_ptr).ref_count;
      return synSuccess;
    }

    // In case of small allocations preallocate larger buffer at once.
    if (size <= size_1mb) {
      actual_allocation_size = size_2mb;
      small_alloc_optimization_enable = true;
    }
  }

  // Allocate a new block if no cached allocation is found.
  auto [err, is_huge_page] = alloc_memory(actual_allocation_size, ptr);
  if (err == synOutOfHostMemory) {
    // Release the cache and retry malloc if the error is OOM.
    PT_SYNHELPER_WARN(
        "SynHostMalloc Failed OOM, Retrying by dropping cache.", err);
    dropCache();
    std::tie(err, is_huge_page) = alloc_memory(actual_allocation_size, ptr);
  }
  if (err != synSuccess) {
    PT_SYNHELPER_WARN("Host memory allocation failed ", err);
    *ptr = nullptr;
    return err;
  }

  if (small_alloc_optimization_enable) {
    // TODO: This sizes could be better selected, depending on the workload.
    // Either some adaptive allocation or some heuristic if hitting large buffer
    // for small allocation.
    constexpr static std::array<size_t, 25> block_sizes = {
        1048576ull, 524288ull, 262144ull, 131072ull, 65536ull,
        32768ull,   16384ull,  8192ull,   4096ull,   2048ull,
        1024ull,    512ull,    256ull,    128ull,    64ull,
        8ull,       8ull,      8ull,      8ull,      8ull,
        8ull,       4ull,      4ull,      4ull,      4ull,
    };
    uint8_t* next_ptr = static_cast<uint8_t*>(*ptr);
    for (const auto block_size : block_sizes) {
      auto [block_it, inserted] = blocks_.insert(
          {next_ptr,
           Block{
               block_size,
               next_ptr,
               Block::is_allocated_t{false},
               is_huge_page,
               Block::is_real_allocation_t{next_ptr == *ptr},
               actual_allocation_size,
               *ptr}});
      available_.insert(block_it->second);
      next_ptr += block_size;
    }

    auto it = available_.lower_bound(BlockSize{size});
    auto& block = blocks_.at(it->ptr);
    block.allocated = Block::is_allocated_t{true};
    *ptr = block.ptr;
    available_.erase(it);
    ++blocks_.at(block.real_allocation_ptr).ref_count;
  } else {
    auto [block_it, inserted] = blocks_.insert(
        {*ptr,
         Block(
             actual_allocation_size,
             *ptr,
             Block::is_allocated_t{true},
             is_huge_page,
             Block::is_real_allocation_t{true},
             actual_allocation_size,
             *ptr)});
    ++block_it->second.ref_count;
  }
  return synSuccess;
}

void host_memory::free_memory(
    void* const ptr,
    Block::is_huge_page_t is_huge_page,
    size_t size) {
  if (is_huge_page) {
    auto err = synHostUnmap(device_.id(), ptr);
    if (err != synSuccess) {
      PT_SYNHELPER_DEBUG("SynHostUnmap Failed.", err);
    }
    auto unmap_err = munmap(ptr, size);
    if (unmap_err != 0) {
      PT_SYNHELPER_DEBUG("Munmap Failed.", unmap_err);
    } else {
      remaining_huge_pages_mb_ += size;
    }
  } else {
    auto err = synHostFree(device_.id(), ptr, 0);
    if (err != synSuccess) {
      // FIXME since the destructor are not called correctly from device
      // call to synHostFree fails.
      PT_SYNHELPER_DEBUG("SynHostFree Failed.", err);
    }
  }
}

synStatus host_memory::free(void* ptr) {
  if (nullptr == ptr) {
    return synSuccess;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = blocks_.find(ptr);
  HABANA_ASSERT(it != blocks_.end());

  Block& block = it->second;
  HABANA_ASSERT(block.allocated);

  block.allocated = Block::is_allocated_t{false};
  --blocks_.at(block.real_allocation_ptr).ref_count;
  if (device_.HostMemoryCacheEnabled()) {
    available_.insert(block);
  } else {
    free_memory(ptr, block.is_huge_page, block.real_allocation_size);
    blocks_.erase(it);
  }
  return synSuccess;
}

synStatus host_memory::uncached_malloc(void** ptr, size_t size) {
  *ptr = nullptr;
  if (size == 0) {
    return synSuccess;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  // Allocate a new block.
  auto [err, is_huge_page] = alloc_memory(size, ptr);
  if (err == synOutOfHostMemory) {
    // Release the cache and retry malloc if the error is OOM.
    PT_SYNHELPER_WARN(
        "SynHostMalloc Failed OOM, Retrying by dropping cache.", err);
    dropCache();
    std::tie(err, is_huge_page) = alloc_memory(size, ptr);
  }
  if (err != synSuccess) {
    *ptr = nullptr;
    return err;
  }

  auto [block_it, inserted] = blocks_.insert(
      {*ptr,
       Block(
           size,
           *ptr,
           Block::is_allocated_t{true},
           is_huge_page,
           Block::is_real_allocation_t{true},
           size,
           *ptr)});
  ++block_it->second.ref_count;
  return synSuccess;
}

synStatus host_memory::uncached_free(void* ptr) {
  if (nullptr == ptr) {
    return synSuccess;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = blocks_.find(ptr);
  HABANA_ASSERT(it != blocks_.end());

  Block& block = it->second;
  HABANA_ASSERT(block.allocated);

  block.allocated = Block::is_allocated_t{false};
  --blocks_.at(block.real_allocation_ptr).ref_count;
  free_memory(ptr, block.is_huge_page, block.real_allocation_size);
  blocks_.erase(it);
  return synSuccess;
}

void host_memory::dropCache() {
  // Free and erase non-allocated blocks.
  for (auto it = blocks_.begin(); it != blocks_.end();) {
    Block& block = it->second;
    if (!block.allocated && block.real_allocation && block.ref_count == 0) {
      // If non-allocated real allocation with no allocated unreal allocations
      // referring it.
      free_memory(block.ptr, block.is_huge_page, block.real_allocation_size);
      available_.erase(block);
      it = blocks_.erase(it);
    } else if (
        !block.allocated && !block.real_allocation &&
        (blocks_.count(block.real_allocation_ptr) == 0 ||
         blocks_.at(block.real_allocation_ptr).ref_count == 0)) {
      // If non-allocated unreal allocation with real allocation already freed
      // or without any unreal allocations referring it.
      available_.erase(block);
      it = blocks_.erase(it);
    } else {
      // If the block is allocated or the ref count is not zero or unreal
      // allocation with real allocation in use.
      ++it;
    }
  }
}

bool host_memory::is_host_memory(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ptr) {
    return false;
  }

  auto it = blocks_.find(ptr);
  if (it == blocks_.end()) {
    return false;
  } else {
    Block& block = it->second;
    if (block.allocated)
      return true;
    else
      return false;
  }
}
} // namespace synapse_helpers
