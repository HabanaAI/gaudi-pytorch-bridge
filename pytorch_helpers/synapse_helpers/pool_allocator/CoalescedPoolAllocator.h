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
#include "PoolAllocator.h"

namespace synapse_helpers {
namespace pool_allocator {

/// bump pooling ///

struct Chunk {
  uint64_t size;
  uint64_t extra_space;
  bool used;
  Chunk* prev;
  Chunk* next;
  uint64_t memptr;
};

struct simple_coalesced_pool_t {
  uint64_t next;
  uint64_t end;
  Chunk* start;
  Chunk* top;
  uint64_t memptr;
  uint64_t basememptr;
};

class StaticCoalescedPooling : public PoolingStrategy {
 private:
  mutable std::list<Chunk*> pool_list;
  mutable std::list<Chunk*> free_list;
  mutable synDeviceId pool_id;
  mutable uint64_t max_pool_size;
  mutable uint64_t chunk_count;
  mutable uint64_t allocted_chunk_size;
  mutable uint64_t free_chunks;
  mutable uint64_t free_chunks_size;
  mutable simple_coalesced_pool_t* prealloc_pool;
  Chunk* reuse_chunks(uint64_t size) const;
  void* get_free_chunk(uint64_t size) const;
  Chunk* get_any_available_free_chunk(uint64_t size) const;
  bool skip_chunk(Chunk* chunk, uint64_t size_req) const;
  bool canMergePreviousChunk(Chunk* chunk, uint64_t size) const;
  Chunk* mergePreviousChunk(Chunk* chunk) const;
  bool canMergeNextChunk(Chunk* chunk, uint64_t size) const;
  Chunk* mergeNextChunk(Chunk* chunk) const;
  Chunk* try_coalescing_chunks(void* ptr, uint64_t size) const;
  Chunk* try_splitting_chunks(void* ptr, uint64_t size) const;
  bool pool_defragment(uint64_t size) const;
  Chunk* create_chunk() const;
  Chunk* try_block_splitting(uint64_t size) const;
  Chunk* try_defragmenting(void* ptr, uint64_t size) const;
  bool isContigousBlockAvailable(uint64_t size) const;
  bool isChunkContigous(Chunk* chunk1, Chunk* chunk2) const;
  uint64_t getContigousChunkSize(Chunk* chunk) const;
  Chunk* defragment_on_reuse(void* ptr, uint64_t size) const;
  void print_pool_stats() const;
  mutable std::recursive_mutex sp_mutex;

 public:
  StaticCoalescedPooling();
  bool pool_create(synDeviceId deviceID, uint64_t size) const override;
  void pool_destroy() const override;
  void* pool_alloc_chunk(uint64_t size) const override;
  void pool_free_chunk(void* p) const override;
};

} // namespace pool_allocator
} // namespace synapse_helpers
