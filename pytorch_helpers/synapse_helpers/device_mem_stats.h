/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <absl/strings/str_format.h>
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
  std::string fragmentation_mask;

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
        fragmentation_mask("") {}

  std::string DebugString() const {
    return absl::StrFormat(
        "Pool ID:           %20lld\n"
        "Limit:             %20lld (%.2f GB)\n"
        "InUse:             %20lld (%.2f MB)\n"
        "MaxInUse:          %20lld (%.2f MB)\n"
        "NumAllocs:         %20lld\n"
        "NumFrees:          %20lld\n"
        "ActiveAllocs:      %20lld\n"
        "ScratchMem:        %20lld (%.2f MB)\n"
        "MaxAllocSize:      %20lld (%.2f MB)\n"
        "TotalSystemAllocs: %20lld\n"
        "TotalSystemFrees:  %20lld\n"
        "TotActiveAllocs:   %20lld\n"
        "Fragmentation:     %20lld\n"
        "FragmentationMask: %20s\n",
        this->pool_id,
        this->memory_limit,
        this->memory_limit / (1024 * 1024 * 1024.),
        this->bytes_in_use,
        this->bytes_in_use / (1024 * 1024.),
        this->peak_bytes_in_use,
        this->peak_bytes_in_use / (1024 * 1024.),
        this->num_allocs,
        this->num_frees,
        (int64_t)this->num_allocs - (int64_t)this->num_frees,
        this->scratch_mem_in_use,
        this->scratch_mem_in_use / (1024 * 1024.),
        this->largest_alloc_size,
        this->largest_alloc_size / (1024 * 1024.),
        this->total_allocs,
        this->total_frees,
        (int64_t)this->total_allocs - (int64_t)this->total_frees,
        this->fragmentation_percent,
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
