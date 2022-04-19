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
#include <synapse_helpers/device_mem_stats.h>
#include <mutex>
#include "synapse_helpers/util.h"

namespace synapse_helpers {
namespace pool_allocator {

enum PoolStrategyType {
  strategy_none = 0,
  strategy_bump,
  strategy_dynamic,
  startegy_static_coalesce,
  startegy_static_coalesce_with_memthreshold,
  startegy_coalesce_stringent,
};

// [Fix Me:] need to have the pool size to accomodate one
// complete model for static pooling
#define DEFAULT_POOL_SIZE 24ULL * 1024 * 1024 * 1024 // 24GByte
#define POOLING_TYPE strategy_bump
#define DEFAULT_ALIGNMENT 128

#define allocateHostMemory new
#define freeHostMemory delete

class PoolingStrategy {
 public:
  virtual ~PoolingStrategy() = default;
  virtual bool pool_create(synDeviceId deviceID, uint64_t size) const = 0;
  virtual void pool_destroy() const = 0;
  virtual void* pool_alloc_chunk(uint64_t size, bool is_workspace = false)
      const = 0;
  virtual void pool_free_chunk(void* p) const = 0;
  virtual bool is_mem_threshold_hit() const = 0;
  virtual void* extend_high_memory_allocation(uint64_t size) const = 0;
  virtual void get_stats(MemoryStats* stats) const = 0;
  virtual void clear_stats() const = 0;
  virtual void reset_peak_mem_stats() const = 0;
  virtual size_t allocated_size(UNUSED const void* p) const {
    return 0;
  }
  virtual std::vector<std::pair<void*, size_t>> get_memory_info() const {
    return {};
  }
  virtual std::pair<void*, size_t> get_tail_chunk_info() const {
    return {};
  }
  virtual std::tuple<void*, size_t, size_t> get_small_alloc_info() const {
    return {};
  }
  virtual bool is_memory_available(UNUSED size_t size) const {
    return true;
  }
};

class SubAllocator {
 private:
  PoolingStrategy* strategy_;

 public:
  SubAllocator(PoolingStrategy* strategy = nullptr) : strategy_(strategy) {}

  ~SubAllocator() {
    delete this->strategy_;
  }

  void set_strategy(PoolingStrategy* strategy) {
    delete this->strategy_;
    this->strategy_ = strategy;
  }

  bool pool_create(synDeviceId deviceID, uint64_t size) const {
    return this->strategy_->pool_create(deviceID, size);
  }

  void pool_destroy() const {
    return this->strategy_->pool_destroy();
  }

  void* pool_alloc_chunk(uint64_t size, bool is_workspace) const {
    return this->strategy_->pool_alloc_chunk(size, is_workspace);
  }

  void pool_free_chunk(void* p) const {
    return this->strategy_->pool_free_chunk(p);
  }

  bool is_mem_threshold_hit() const {
    return this->strategy_->is_mem_threshold_hit();
  }

  void* extend_high_memory_allocation(uint64_t size) const {
    return this->strategy_->extend_high_memory_allocation(size);
  }

  void get_stats(MemoryStats* stats) const {
    this->strategy_->get_stats(stats);
  }

  void clear_stats() const {
    this->strategy_->clear_stats();
  }

  void reset_peak_mem_stats() const {
    this->strategy_->reset_peak_mem_stats();
  }

  size_t allocated_size(const void* p) const {
    return this->strategy_->allocated_size(p);
  }

  std::vector<std::pair<void*, size_t>> get_memory_info() const {
    return this->strategy_->get_memory_info();
  }

  std::pair<void*, size_t> get_tail_chunk_info() const {
    return this->strategy_->get_tail_chunk_info();
  }

  std::tuple<void*, size_t, size_t> get_small_alloc_info() const {
    return this->strategy_->get_small_alloc_info();
  }

  bool is_memory_available(size_t size) const {
    return this->strategy_->is_memory_available(size);
  }
};

/// bump pooling ///

struct Poolchunk {
  uint64_t size;
  bool used;
  Poolchunk* next;
  uint64_t memptr;
};

struct simple_pool_t {
  uint64_t next;
  uint64_t end;
  Poolchunk* _start;
  Poolchunk* _top;
  uint64_t memptr;
};

class StaticPooling : public PoolingStrategy {
 private:
  mutable synDeviceId pool_id;
  mutable uint64_t max_pool_size;
  mutable uint64_t block_count;
  mutable uint64_t allocted_block_size;
  mutable uint64_t free_chunks;
  mutable uint64_t free_chunks_size;
  mutable uint64_t bytes_in_use;
  mutable MemoryStats stats;
  mutable simple_pool_t* prealloc_pool;
  void* reuse_chunks(uint64_t size) const;
  void* get_free_chunk(void* p, uint64_t size) const;
  void print_pool_stats() const;
  mutable std::mutex sp_mutex;

 public:
  StaticPooling();
  bool pool_create(synDeviceId deviceID, uint64_t size) const override;
  void pool_destroy() const override;
  void* pool_alloc_chunk(uint64_t size, bool is_workspace) const override;
  void pool_free_chunk(void* p) const override;
  bool is_mem_threshold_hit() const override;
  void* extend_high_memory_allocation(uint64_t size) const override;
  void get_stats(MemoryStats* stats) const override;
  void clear_stats() const override;
  void reset_peak_mem_stats() const override;
};

/// Variable length pooling using equal fit block ///

struct Block {
  uint64_t size;
  bool used;
  Block* next;
  uint64_t memptr;
};

class DynamicPooling : public PoolingStrategy {
 private:
  mutable synDeviceId pool_id;
  mutable Block* pool_start;
  mutable Block* top;
  mutable uint64_t bytes_in_use;
  mutable MemoryStats stats;
  Block* retrieveBlock(void* data) const;
  Block* requestNewBlock(uint64_t size) const;
  Block* equalFit(uint64_t size) const;
  Block* findBlock(uint64_t size) const;
  void* allocBlock(uint64_t size) const;
  void freeBlock(void* data) const;
  void freeBlocks(Block* base_block) const;
  void freeUnusedBlocks(Block* base_block) const;
  mutable std::mutex vp_mutex;

 public:
  DynamicPooling();
  bool pool_create(synDeviceId deviceID, uint64_t size) const override;
  void pool_destroy() const override;
  void* pool_alloc_chunk(uint64_t size, bool is_workspace) const override;
  void pool_free_chunk(void* p) const override;
  bool is_mem_threshold_hit() const override;
  void* extend_high_memory_allocation(uint64_t size) const override;
  void get_stats(MemoryStats* stats) const override;
  void clear_stats() const override;
  void reset_peak_mem_stats() const override;
};

} // namespace pool_allocator
} // namespace synapse_helpers
