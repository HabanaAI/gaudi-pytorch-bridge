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
#include <absl/strings/str_format.h>
#include <synapse_api.h>
namespace synapse_helpers {
struct MemoryStats {
  synDeviceId pool_id;
  uint64_t num_allocs; /* Number of allocs from start_collect to stop_collect.*/
  uint64_t total_allocs; /* Total Number of allocations.*/
  uint64_t bytes_in_use; /* Number of bytes in use. */
  uint64_t peak_bytes_in_use; /* The maximum bytes in use. */
  uint64_t largest_alloc_size; /* The largest single allocation seen */
  uint64_t num_frees; /* Number of frees from start_collect to stop_collect.*/
  uint64_t total_frees; /* Total number of frees.*/
  uint64_t memory_limit; /* Max memory bytes */
  uint64_t scratch_mem_in_use; /* internal memory used */
  uint64_t
      fragmentation_percent; /* fragmentation % = 100 x (1-
                                max_contiguous_free_chunk/total_free_chunk_memory)
                              */

  uint64_t total_chunks;
  uint64_t total_size;
  uint64_t occupied_chunks;
  uint64_t occupied_size;
  uint64_t free_chunks;
  uint64_t free_chunks_size;
  uint64_t max_cntgs_free_chunks_size;
  uint64_t total_extra_spaced_chunks;
  uint64_t total_extra_size;
  std::string fragmentation_mask;

  uint64_t pre_allocate_size;
  uint64_t min_chunk_size;
  uint64_t max_chunk_size;

  MemoryStats()
      : num_allocs(0),
        total_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        num_frees(0),
        total_frees(0),
        memory_limit(0),
        scratch_mem_in_use(0),
        fragmentation_percent(0),
        total_chunks(0),
        total_size(0),
        occupied_chunks(0),
        occupied_size(0),
        free_chunks(0),
        free_chunks_size(0),
        max_cntgs_free_chunks_size(0),
        total_extra_spaced_chunks(0),
        total_extra_size(0),
        fragmentation_mask(""),
        pre_allocate_size(0),
        min_chunk_size(0),
        max_chunk_size(0) {}

  std::string DebugString() const {
    return absl::StrFormat(
        "Pool ID:                     %20lld\n"
        "Limit:                       %20lld (%.2f GB)\n"
        "InUse:                       %20lld (%.2f MB)\n"
        "MaxInUse:                    %20lld (%.2f MB)\n"
        "NumAllocs:                   %20lld\n"
        "NumFrees:                    %20lld\n"
        "ActiveAllocs:                %20lld\n"
        "ScratchMem:                  %20lld (%.2f MB)\n"
        "MaxAllocSize:                %20lld (%.2f MB)\n"
        "TotalSystemAllocs:           %20lld\n"
        "TotalSystemFrees:            %20lld\n"
        "TotActiveAllocs:             %20lld\n"
        "Fragmentation:               %20lld\n"
        "total_chunks:                %20lld\n"
        "total_size:                  %20lld (%.2f MB)\n"
        "occupied_chunks:             %20lld\n"
        "occupied_size:               %20lld (%.2f MB)\n"
        "free_chunks:                 %20lld\n"
        "free_chunks_size:            %20lld (%.2f MB)\n"
        "max_cntgs_free_size:         %20lld (%.2f MB)\n"
        "total_extra_spaced_chunks:   %20lld\n"
        "total_extra_size:            %20lld (%.2f MB)\n"
        "FragmentationMask: %20s\n",
        this->pool_id,
        this->memory_limit,
        static_cast<double>(this->memory_limit) / (1024 * 1024 * 1024.),
        this->bytes_in_use,
        static_cast<double>(this->bytes_in_use) / (1024 * 1024.),
        this->peak_bytes_in_use,
        static_cast<double>(this->peak_bytes_in_use) / (1024 * 1024.),
        this->num_allocs,
        this->num_frees,
        (int64_t)this->num_allocs - (int64_t)this->num_frees,
        this->scratch_mem_in_use,
        static_cast<double>(this->scratch_mem_in_use) / (1024 * 1024.),
        this->largest_alloc_size,
        static_cast<double>(this->largest_alloc_size) / (1024 * 1024.),
        this->total_allocs,
        this->total_frees,
        (int64_t)this->total_allocs - (int64_t)this->total_frees,
        this->fragmentation_percent,
        this->total_chunks,
        this->total_size,
        static_cast<double>(this->total_size) / (1024 * 1024.),
        this->occupied_chunks,
        this->occupied_size,
        static_cast<double>(this->occupied_size) / (1024 * 1024.),
        this->free_chunks,
        this->free_chunks_size,
        static_cast<double>(this->free_chunks_size) / (1024 * 1024.),
        this->max_cntgs_free_chunks_size,
        static_cast<double>(this->max_cntgs_free_chunks_size) / (1024 * 1024.),
        this->total_extra_spaced_chunks,
        this->total_extra_size,
        static_cast<double>(this->total_extra_size) / (1024 * 1024.),
        this->fragmentation_mask);
  };

  void UpdateStats(uint64_t size, bool is_alloc, bool is_workspace = false) {
    if (is_alloc) {
      ++this->num_allocs;
      ++this->total_allocs;
      this->bytes_in_use += size;
      this->peak_bytes_in_use =
          std::max<uint64_t>(this->peak_bytes_in_use, this->bytes_in_use);
      this->largest_alloc_size =
          std::max<uint64_t>(this->largest_alloc_size, size);
      if (is_workspace)
        this->scratch_mem_in_use = size;

    } else {
      ++this->num_frees;
      ++this->total_frees;
      this->bytes_in_use -= size;
    }
  }
};
} // namespace synapse_helpers
