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
#include <synapse_api_types.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <set>
#include <unordered_map>
#include "Chunk.h"
#include "PoolAllocator.h"
#include "backend/synapse_helpers/RealTimeMemoryLogger.h"
#include "backend/synapse_helpers/util.h"
#include "utils.h"

namespace synapse_helpers {
namespace pool_allocator {

static const uint64_t kInvalidBinNum = -1;
// The largest bin'd chunk size is 256 << 21 = 512MB.
static const uint64_t kNumBins = 21;

// Bin: collection of similar-sized free chunks.
struct Bin {
  // All chunks in this bin have >= bin_size memory.
  size_t bin_size = 0;

  struct chunkcompare {
    bool operator()(const Chunk* a, const Chunk* b) const {
      // sort by size, break ties with pointer
      if (a->size != b->size) {
        return a->size < b->size;
      }
      return a->memptr < b->memptr;
    };
  };

  using FreeChunkSet = std::set<Chunk*, chunkcompare>;
  // List of free chunks within the bin, sorted by chunk size.
  FreeChunkSet free_chunks;
  Bin(size_t bs) : bin_size(bs), free_chunks(chunkcompare()) {}
};

class BinUtils {
  std::array<char, sizeof(Bin) * kNumBins> bins_space;
  size_t kMinAllocationSize;
  size_t kMinAllocationBits;
  inline uint64_t Log2FloorNonZero(uint64_t n) const {
    uint64_t r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

 public:
  BinUtils(size_t minAllocationsize, size_t minAllocationBits)
      : kMinAllocationSize{minAllocationsize},
        kMinAllocationBits{minAllocationBits} {}

  // Map from bin size to Bin
  Bin* BinFromIndex(uint64_t index) const;
  size_t BinNumToSize(uint64_t index) {
    return kMinAllocationSize << index; /* kMinAllocationSize = 1 << 8 = 256 */
  }
  uint64_t BinIndexForSize(size_t bytes) const;
  Bin* BinForSize(size_t bytes) const;
  void InsertFreeChunkIntoBin(Chunk* c) const;
  void RemoveFreeChunkFromBin(Chunk* c) const;
  void RemoveFreeChunkIterFromBin(
      Bin::FreeChunkSet* free_chunks,
      const Bin::FreeChunkSet::iterator& citer) const;
};

class CoalescedStringentPooling : public PoolingStrategy {
 public:
  CoalescedStringentPooling(device& device);
  ~CoalescedStringentPooling();
  bool pool_create(synDeviceId deviceID, uint64_t size) const override;
  void pool_destroy() const override;
  void* pool_alloc_chunk(uint64_t size, [[maybe_unused]] bool is_workspace)
      const override;
  void* pool_alloc_chunk(
      uint64_t size,
      hpuStream_t stream,
      bool use_stream = false) const override;
  void pool_free_chunk(void* p) const override;
  void* extend_high_memory_allocation(uint64_t size, size_t current_ws_size)
      const override;
  void get_stats(MemoryStats* stats) const override;
  std::vector<std::pair<uint64_t, uint64_t>> get_occupied_chunk_map()
      const override;
  void clear_stats() const override;
  void reset_peak_mem_stats() const override;
  size_t allocated_size(const void* ptr) const override;
  std::vector<std::pair<void*, size_t>> get_memory_info() const override;
  std::pair<void*, size_t> get_tail_chunk_info() const override;
  std::tuple<void*, size_t, size_t> get_small_alloc_info() const override;
  bool is_memory_available(size_t size) const override;
  bool is_memory_available(
      size_t persistant_size,
      size_t curr_ws_size,
      size_t new_ws_size) const override;
  void print_pool_stats() const override;
  size_t get_max_cntgs_chunk_size() const override;
  void set_defragmenter_state(bool started) const override;
  void record_stream(void* ptr, hpuStream_t stream) const override;
  bool is_stream_uses_empty(void* p) const override;
  void synchronize_and_free_events() const override;

  void get_memory_mask(std::vector<uint64_t>& mmask) const;

 private:
  mutable std::unique_ptr<realtime_logger::RealTimeMeoryLogger>
      realtime_logger_;
  struct chunkcompare {
    bool operator()(const Chunk* a, const Chunk* b) {
      // sort by memptr
      return a->memptr < b->memptr;
    };
  };
  mutable std::unordered_map<uint64_t, Chunk*> chunks;
  mutable synDeviceId pool_id;
  mutable uint64_t max_pool_size;
  mutable uint64_t chunk_count;
  mutable uint64_t allocted_chunk_size;
  mutable uint64_t free_chunks;
  mutable uint64_t free_chunks_size;
  mutable uint64_t bytes_in_use;
  mutable simple_coalesced_pool_t* prealloc_pool;
  mutable BinUtils* bin_utils;
  mutable bool high_memory_allocated_ = false;
  mutable bool defragmenter_state_started_ = false;
  mutable MemoryStats stats;
  device& device_;
  mutable size_t alignment;
  mutable size_t kMinAllocationBits;
  mutable size_t kMinAllocationSize;
  size_t header_bytes;

  void* alloc_chunk(uint64_t size, hpuStream_t stream, bool use_stream = false)
      const;
  void delete_chunk(void* p) const;
  void delete_chunk(void* p, hpuStream_t stream) const;
  Chunk* reuse_chunks(
      uint64_t size,
      hpuStream_t stream,
      bool use_stream = false) const;
  void try_splitting_chunks(Chunk* chunk, uint64_t size) const;
  Chunk* try_to_merge(Chunk* c) const;
  void merge(Chunk* c1, Chunk* c2) const;
  Chunk* create_chunk() const;
  Chunk* try_block_splitting(uint64_t size) const;
  bool isChunkContigous(Chunk* chunk1, Chunk* chunk2) const;
  uint64_t getContigousChunkSize(Chunk* chunk) const;
  mutable std::mutex sp_mutex;

  void* FindChunkPtr(
      uint64_t bin_index,
      size_t num_bytes,
      hpuStream_t stream,
      bool use_stream = false) const;
  void insert_events(Chunk* chunk) const;
  void process_events(void) const;

  class SmallAllocs {
   public:
    SmallAllocs() = delete;
    SmallAllocs(
        std::unique_ptr<int8_t, std::function<void(int8_t*)>>,
        size_t,
        size_t,
        size_t);
    ~SmallAllocs();
    SmallAllocs(SmallAllocs&& rhs) noexcept;
    SmallAllocs& operator=(SmallAllocs&& rhs) noexcept;
    bool IsAllocated(const void* ptr) const;
    size_t Size(const void* ptr) const;
    void* Allocate(size_t num_bytes);
    void Deallocate(const void* ptr);
    void Reset();
    size_t UnitsOccupied() const;
    void* GetChunkPtr();
    size_t GetkSize() {
      return kSize_;
    }
    size_t GetkThreshold() {
      return kThreshold_;
    }

   private:
    void ValidateEmpty() const;
    size_t Offset(const void* ptr) const;
    size_t ToUnits(size_t offset_in_bytes) const;
    size_t ToBytes(size_t offset_in_units) const;

    std::unique_ptr<int8_t, std::function<void(int8_t*)>> chunk_ptr_;
    std::vector<bool> map_;
    std::vector<size_t> size_;
    std::size_t kAlignment_;
    std::size_t kSize_;
    std::size_t kThreshold_;
    std::size_t kUnits_;
  };

  mutable std::unique_ptr<SmallAllocs> small_allocs_;
  mutable std::
      unordered_map<hpuStream_t, std::deque<std::pair<synEventHandle, Chunk*>>>
          hpu_events;
};
} // namespace pool_allocator
} // namespace synapse_helpers
