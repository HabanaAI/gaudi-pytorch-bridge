/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <list>
#include <set>
#include <unordered_map>
#include "Chunk.h"
#include "PoolAllocator.h"

namespace synapse_helpers {
namespace pool_allocator {

class StaticCoalescedPooling : public PoolingStrategy {
 private:
  struct chunkcompare {
    bool operator()(const Chunk* a, const Chunk* b) const {
      // sort by size, break ties with pointer
      if (a->size != b->size) {
        return a->size < b->size;
      }
      return a->memptr < b->memptr;
    };
  };
  mutable std::set<Chunk*, chunkcompare> free_list;
  mutable std::unordered_map<uint64_t, Chunk*> chunks;
  mutable synDeviceId pool_id;
  mutable uint64_t max_pool_size;
  mutable uint64_t chunk_count;
  mutable uint64_t allocted_chunk_size;
  mutable uint64_t free_chunks;
  mutable uint64_t free_chunks_size;
  mutable uint64_t bytes_in_use;
  mutable MemoryStats stats;
  mutable simple_coalesced_pool_t* prealloc_pool;
  Chunk* reuse_chunks(uint64_t size) const;
  Chunk* get_free_chunk(uint64_t size) const;
  Chunk* get_any_available_free_chunk(uint64_t size) const;
  Chunk* get_nearest_chunk(uint64_t size) const;
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
  Chunk* defragment_on_reuse(void* ptr, uint64_t size) const;
  mutable std::mutex sp_mutex;

 public:
  StaticCoalescedPooling();
  bool pool_create(synDeviceId deviceID, uint64_t size) const override;
  void pool_destroy() const override;
  void* pool_alloc_chunk(uint64_t size, bool is_workspace) const override;
  void pool_free_chunk(void* p) const override;
  bool is_mem_threshold_hit() const override;
  void* extend_high_memory_allocation(uint64_t size, size_t curr_ws)
      const override;
  void get_stats(MemoryStats* stats) const override;
  std::vector<std::pair<uint64_t, uint64_t>> get_occupied_chunk_map()
      const override;
  void clear_stats() const override;
  void reset_peak_mem_stats() const override;
  void print_pool_stats() const override;
};

} // namespace pool_allocator
} // namespace synapse_helpers
