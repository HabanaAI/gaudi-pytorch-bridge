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

#define TO_GB(arg) ((arg) / (1024 * 1024 * 1024.))
#define TO_REPORT_EVENT(key, value)                                         \
  std::string(" \"") + key + std::string("\":\"") + std::to_string(value) + \
      std::string("\"")
#define TO_REPORT_EVENT_GB(key, value)                                      \
  std::string(" \"") + key + std::string("\":\"") + std::to_string(value) + \
      std::string(" (") + std::to_string((value) / (1024 * 1024 * 1024.)) + \
      std::string(" GB)\"")

namespace synapse_helpers {
struct MemoryConsumption {
  uint64_t total_allocs_bytes; /* Total of bytes allocated. */
  uint64_t max_alloc_bytes; /* The maximum single byte allocated. */
  uint64_t
      pre_allocated_bytes; /* Preallocate bytes allocated for HCL or other. */
  uint64_t workspace_allocated; /* Scratch memory allocated. */
  uint64_t persistent_tensor_size; /* Persistent memory allocated */
  // uint64_t live_tensors_allocs_bytes; /* Live tensors bytes in use. */
  // uint64_t ghost_tensors_allocs_bytes; /* Ghost tensors bytes in use. */

  MemoryConsumption()
      : total_allocs_bytes(0),
        max_alloc_bytes(0),
        pre_allocated_bytes(0),
        workspace_allocated(0),
        persistent_tensor_size(0) {}

  std::string DebugString() const {
    return absl::StrFormat(
        "Total Allocate Size:               %20lld (%.6f GB)\n"
        "Maximum Allocate Size:             %20lld (%.6f GB)\n"
        "Pre Allocate Size:                 %20lld (%.6f GB)\n"
        "Scratch Memory Allocated:          %20lld (%.6f GB)\n"
        "Persistent Memory Allocated:       %20lld (%.6f GB)\n",
        this->total_allocs_bytes,
        TO_GB(this->total_allocs_bytes),
        this->max_alloc_bytes,
        TO_GB(this->max_alloc_bytes),
        this->pre_allocated_bytes,
        TO_GB(this->pre_allocated_bytes),
        this->workspace_allocated,
        TO_GB(this->workspace_allocated),
        this->persistent_tensor_size,
        TO_GB(this->persistent_tensor_size));
  };

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") + "MemoryConsumption\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"MemoryConsumption\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT_GB("TotalAllocatedSize", this->total_allocs_bytes) +
        std::string(",") +
        TO_REPORT_EVENT_GB("PreAllocatedSize", this->pre_allocated_bytes) +
        std::string(",") +
        TO_REPORT_EVENT_GB("ScratchMemoryAllocateSize",
                           this->workspace_allocated) +
        std::string(",") +
        TO_REPORT_EVENT_GB("PersistentMemoryAllocateSize",
                           this->persistent_tensor_size) +
        std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "MemoryConsumption\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"MemoryConsumption\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};

struct MemoryAllocatorStats {
  uint64_t total_num_allocs; /* Total number of allocations. */
  uint64_t new_num_allocs; /* New number of allocations. */
  uint64_t total_num_frees; /* Total number of frees. */
  uint64_t new_num_frees; /* New number of frees. */

  MemoryAllocatorStats()
      : total_num_allocs(0),
        new_num_allocs(0),
        total_num_frees(0),
        new_num_frees(0) {}

  std::string DebugString() const {
    return absl::StrFormat(
        "Total Number of Allocs:            %20lld\n"
        "New Number of Allocs:              %20lld\n"
        "Total Number of Frees:             %20lld\n"
        "New Number of Frees:               %20lld\n",
        this->total_num_allocs,
        this->new_num_allocs,
        this->total_num_frees,
        this->new_num_frees);
  };

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") +
        "MemoryAllocatorStats\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"MemoryAllocatorStats\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT("TotalNumAllocs", this->total_num_allocs) +
        std::string(",") +
        TO_REPORT_EVENT("NewNumAllocs", this->new_num_allocs) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumFrees", this->total_num_frees) +
        std::string(",") + TO_REPORT_EVENT("NewNumFrees", this->new_num_frees) +
        std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "MemoryAllocatorStats\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"MemoryAllocatorStats\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};

struct FragmentationStats {
  uint64_t fragmentation_percent; /* Fragmentation percentage. */
  uint64_t total_num_chunks; /* Total number of chunks. */
  uint64_t total_num_alloc_chunks; /* Total number of alloc chunks. */
  uint64_t total_num_free_chunks; /* Total number of free chunks. */
  uint64_t total_alloc_size; /* Total alloc size. */
  uint64_t total_free_size; /* Total free size. */
  uint64_t max_cntg_chunk_free_size; /* Maximum contiguous chunk free size
                                        available. */
  uint64_t min_chunk_size; /* Minimum chunk size. */
  uint64_t max_chunk_size; /* Maximum Chunk size. */
  std::string fragmentation_histogram; /* Fragmentation histogram. */

  FragmentationStats()
      : fragmentation_percent(0),
        total_num_chunks(0),
        total_num_alloc_chunks(0),
        total_num_free_chunks(0),
        total_alloc_size(0),
        total_free_size(0),
        max_cntg_chunk_free_size(0),
        min_chunk_size(0),
        max_chunk_size(0),
        fragmentation_histogram("") {}

  std::string DebugString() const {
    return absl::StrFormat(
        "Fragmentation Percentage:            %20lld\n"
        "Total Number of Chunks:              %20lld\n"
        "Total Number of Alloc Chunks:        %20lld\n"
        "Total Number of Free Chunks:         %20lld\n"
        "Total Alloc Size:                    %20lld\n"
        "Total Free Size:                     %20lld\n"
        "Max Contiguous Chunk Free Size:      %20lld\n"
        "Minimum Chunk size:                  %20lld\n"
        "Maximum Chunk size:                  %20lld\n"
        "Fragmentation Histogram:             %20s\n",
        this->fragmentation_percent,
        this->total_num_chunks,
        this->total_num_alloc_chunks,
        this->total_num_free_chunks,
        this->total_alloc_size,
        this->total_free_size,
        this->max_cntg_chunk_free_size,
        this->min_chunk_size,
        this->max_chunk_size,
        this->fragmentation_histogram);
  };

  std::string toJsonEvent(std::string& header_begin, std::string& header_end)
      const {
    std::string event_chunk_begin = header_begin;
    event_chunk_begin += std::string(", \"name\":\"") + "FragmentationStats\"" +
        std::string(", \"ph\":\"B\", \"cat\":\"FragmentationStats\"") +
        std::string(", \"args\": {") +
        TO_REPORT_EVENT("FragmentationPercentage",
                        this->fragmentation_percent) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumChunks", this->total_num_chunks) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumAllocChunks", this->total_num_alloc_chunks) +
        std::string(",") +
        TO_REPORT_EVENT("TotalNumFreeChunks", this->total_num_free_chunks) +
        std::string(",") +
        TO_REPORT_EVENT_GB("TotalAllocSize", this->total_alloc_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("TotalFreeSize", this->total_free_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("MaxContiguousFreeSize",
                           this->max_cntg_chunk_free_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("MinChunkSize", this->min_chunk_size) +
        std::string(",") +
        TO_REPORT_EVENT_GB("MaxChunkSize", this->max_chunk_size) +
        std::string(",") + std::string(" \"FragmentationHistogram\":\"") +
        this->fragmentation_histogram + std::string("\"") +
        std::string("}},\n");
    std::string event_chunk_end = header_end;
    event_chunk_end += std::string(", \"name\":\"") + "FragmentationStats\"" +
        std::string(", \"ph\":\"E\", \"cat\":\"FragmentationStats\"") +
        std::string("},\n");
    return absl::StrFormat("%s%s", event_chunk_begin, event_chunk_end);
  };
};
} // namespace synapse_helpers