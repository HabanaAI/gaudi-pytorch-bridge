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
#include <synapse_api_types.h>
#include <mutex>

#include "backend/synapse_helpers/device_mem_stats.h"
#include "backend/synapse_helpers/device_types.h"

namespace synapse_helpers {
namespace pool_allocator {

enum PoolStrategyType {
  strategy_none = 0,
  strategy_bump_unused, // unused. depends on "PT_HPU_POOL_STRATEGY" handling.
  strategy_dynamic_unused, // unused. depends on "PT_HPU_POOL_STRATEGY"
  startegy_static_coalesce_unused, // unused. depends on "PT_HPU_POOL_STRATEGY"
  startegy_coalesce_stringent,
};

class PoolingStrategy {
 public:
  virtual ~PoolingStrategy() = default;
  virtual bool pool_create(synDeviceId deviceID, uint64_t size) const = 0;
  virtual void pool_destroy() const = 0;
  virtual void* pool_alloc_chunk(uint64_t size, bool is_workspace = false)
      const = 0;
  virtual void* pool_alloc_chunk(
      [[maybe_unused]] uint64_t size,
      [[maybe_unused]] hpuStream_t stream,
      [[maybe_unused]] bool use_stream = false) const {
    return nullptr;
  }
  virtual void pool_free_chunk(void* p) const = 0;
  virtual void* extend_high_memory_allocation(
      uint64_t size,
      size_t current_ws_size) const = 0;
  virtual void get_stats(MemoryStats* stats) const = 0;
  virtual std::vector<std::pair<uint64_t, uint64_t>> get_occupied_chunk_map()
      const = 0;
  virtual void clear_stats() const = 0;
  virtual void reset_peak_mem_stats() const = 0;
  virtual size_t allocated_size([[maybe_unused]] const void* p) const {
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
  virtual bool is_memory_available([[maybe_unused]] size_t size) const {
    return true;
  }
  virtual void print_pool_stats() const {};
  virtual size_t get_max_cntgs_chunk_size() const {
    return -1;
  };

  virtual bool is_memory_available(
      [[maybe_unused]] size_t persistant_size,
      [[maybe_unused]] size_t curr_ws_size,
      [[maybe_unused]] size_t new_ws_size) const {
    return true;
  }

  virtual void set_defragmenter_state([[maybe_unused]] bool running) const {}

  virtual void record_stream(
      [[maybe_unused]] void* ptr,
      [[maybe_unused]] hpuStream_t stream) const {}
  virtual bool is_stream_uses_empty([[maybe_unused]] void* p) const {
    return true;
  };
  virtual void synchronize_and_free_events() const {};

  // [Fix Me:] need to have the pool size to accomodate one
  // complete model for static pooling
  const unsigned long long int default_pool_size =
      24ULL * 1024 * 1024 * 1024; // 24GByte
};

class SubAllocator {
 private:
  PoolingStrategy* strategy_;

 public:
  SubAllocator(PoolingStrategy* strategy) : strategy_(strategy) {}

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

  void* pool_alloc_chunk(uint64_t size, hpuStream_t stream, bool use_stream)
      const {
    return this->strategy_->pool_alloc_chunk(size, stream, use_stream);
  }

  void pool_free_chunk(void* p) const {
    return this->strategy_->pool_free_chunk(p);
  }

  void* extend_high_memory_allocation(uint64_t size, size_t current_ws_size)
      const {
    return this->strategy_->extend_high_memory_allocation(
        size, current_ws_size);
  }

  std::vector<std::pair<uint64_t, uint64_t>> get_occupied_chunk_map() const {
    return this->strategy_->get_occupied_chunk_map();
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

  void print_pool_stats() const {
    return this->strategy_->print_pool_stats();
  }

  bool is_memory_available(
      size_t persistant_size,
      size_t curr_ws_size,
      size_t new_ws_size) const {
    return this->strategy_->is_memory_available(
        persistant_size, curr_ws_size, new_ws_size);
  }

  size_t get_max_cntgs_chunk_size() const {
    return this->strategy_->get_max_cntgs_chunk_size();
  }

  void set_defragmenter_state(bool started) const {
    this->strategy_->set_defragmenter_state(started);
  }

  void record_stream(void* ptr, hpuStream_t stream) const {
    this->strategy_->record_stream(ptr, stream);
  }

  bool is_stream_uses_empty(void* p) const {
    return this->strategy_->is_stream_uses_empty(p);
  }

  void synchronize_and_free_events() const {
    this->strategy_->synchronize_and_free_events();
  }
};

} // namespace pool_allocator
} // namespace synapse_helpers
