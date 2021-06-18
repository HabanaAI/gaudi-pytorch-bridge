/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
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
#include <list>
#include <unordered_map>
#include "Chunk.h"
#include "PoolAllocator.h"
#include "utils.h"

namespace synapse_helpers {
namespace pool_allocator {

// Bin: collection of similar-sized free chunks.
struct Bin {
  // All chunks in this bin have >= bin_size memory.
  size_t bin_size = 0;

  struct chunkcompare {
    bool operator()(const Chunk* a, const Chunk* b) {
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
 private:
  struct chunkcompare {
    bool operator()(const Chunk* a, const Chunk* b) {
      // sort by size, break ties with pointer
      if (a->size != b->size) {
        return a->size < b->size;
      }
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

  Chunk* reuse_chunks(uint64_t size) const;
  Chunk* get_free_chunk(uint64_t size) const;
  Chunk* get_any_available_free_chunk(uint64_t size) const;
  bool skip_chunk(Chunk* chunk, uint64_t size_req) const;
  bool canMergePreviousChunk(Chunk* chunk, uint64_t size) const;
  bool canMergeNextChunk(Chunk* chunk, uint64_t size) const;
  Chunk* try_splitting_chunks(Chunk* chunk, uint64_t size) const;
  bool pool_defragment(uint64_t size) const;
  bool merge_chunks(std::list<uint64_t> ptrs, bool merge_nxt, uint64_t size)
      const;
  Chunk* merge(Chunk* c1, Chunk* c2) const;
  Chunk* create_chunk() const;
  Chunk* try_block_splitting(uint64_t size) const;
  Chunk* try_defragmenting(void* ptr, uint64_t size) const;
  bool isContigousBlockAvailable(uint64_t size) const;
  bool isChunkContigous(Chunk* chunk1, Chunk* chunk2) const;
  uint64_t getContigousChunkSize(Chunk* chunk) const;
  void print_pool_stats() const;
  mutable std::mutex sp_mutex;

  void* FindChunkPtr(uint64_t bin_index, size_t num_bytes) const;

 public:
  CoalescedStringentPooling();
  bool pool_create(synDeviceId deviceID, uint64_t size) const override;
  void pool_destroy() const override;
  void* pool_alloc_chunk(uint64_t size) const override;
  void pool_free_chunk(void* p) const override;
  bool is_mem_threshold_hit() const override;
};

} // namespace pool_allocator
} // namespace synapse_helpers
