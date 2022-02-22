/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include <synapse_api_types.h>
#include <synapse_helpers/device.h>
#include <atomic>
#include <deque>
#include <list>
#include <unordered_map>
#include "Chunk.h"
#include "PoolAllocator.h"
#include "synapse_helpers/util.h"
#include "utils.h"

namespace synapse_helpers {
namespace pool_allocator {

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
  inline uint64_t Log2FloorNonZero(uint64_t n) const {
    uint64_t r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

 public:
  BinUtils() {}

  // Map from bin size to Bin
  Bin* BinFromIndex(uint64_t index) const;
  static constexpr size_t BinNumToSize(uint64_t index) {
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
  CoalescedStringentPooling(uint64_t max_count, bool enable_merge);
  ~CoalescedStringentPooling();
  bool pool_create(synDeviceId deviceID, uint64_t size) const override;
  void pool_destroy() const override;
  void* pool_alloc_chunk(uint64_t size, UNUSED bool is_workspace)
      const override;
  void pool_free_chunk(void* p) const override;
  bool is_mem_threshold_hit() const override;
  void* extend_high_memory_allocation(uint64_t size) const override;
  void get_stats(MemoryStats* stats) const override;
  void clear_stats() const override;
  size_t allocated_size(const void* ptr) const;
  std::vector<std::pair<void*, size_t>> get_memory_info() const;
  std::pair<void*, size_t> get_tail_chunk_info() const;
  std::tuple<void*, size_t, size_t> get_small_alloc_info() const;

 private:
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
  mutable bool enable_lfu_merging;
  mutable std::deque<Chunk*> chunks_to_merge;
  uint64_t max_merge_count;
  mutable bool high_memory_allocated_ = false;
  mutable MemoryStats stats;
  mutable uint32_t mem_threshold;

  void* alloc_chunk(uint64_t size) const;
  void delete_chunk(void* p) const;
  Chunk* reuse_chunks(uint64_t size) const;
  Chunk* get_free_chunk(uint64_t size) const;
  Chunk* get_any_available_free_chunk(uint64_t size) const;
  void try_splitting_chunks(Chunk* chunk, uint64_t size) const;
  Chunk* try_to_merge(Chunk* c, bool freed_count) const;
  bool defragment_chunks(uint64_t size) const;
  void merge(Chunk* c1, Chunk* c2) const;
  Chunk* create_chunk() const;
  Chunk* try_block_splitting(uint64_t size) const;
  Chunk* try_defragmenting(uint64_t size) const;
  bool isChunkContigous(Chunk* chunk1, Chunk* chunk2) const;
  uint64_t getContigousChunkSize(Chunk* chunk) const;
  void print_pool_stats() const;
  mutable std::mutex sp_mutex;

  void* FindChunkPtr(uint64_t bin_index, size_t num_bytes) const;

  class SmallAllocs {
   public:
    static const std::size_t kAlignment = DEFAULT_ALIGNMENT;
    static const std::size_t kSize = 16 * 1024 * kAlignment;
    static const std::size_t kThreshold = 2 * kAlignment;
    static_assert(kAlignment <= kThreshold, "");
    static_assert(kSize % kAlignment == 0, "kAlignment must divide kSize");
    static const std::size_t kUnits = kSize / kAlignment;

    SmallAllocs() = delete;
    SmallAllocs(std::unique_ptr<int8_t, std::function<void(int8_t*)>>);
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

   private:
    void ValidateEmpty() const;
    size_t Offset(const void* ptr) const;
    static size_t ToUnits(size_t offset_in_bytes);
    static size_t ToBytes(size_t offset_in_units);

    std::unique_ptr<int8_t, std::function<void(int8_t*)>> chunk_ptr_;
    std::vector<bool> map_;
    std::array<size_t, kUnits> size_;
  };

  mutable std::unique_ptr<SmallAllocs> small_allocs_;
};

} // namespace pool_allocator
} // namespace synapse_helpers
