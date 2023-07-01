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
#include <synapse_api.h>

#include <habana_helpers/logging.h>
#include "PoolAllocator.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/devmem_logger.h"
#include "utils.h"

namespace synapse_helpers {
namespace pool_allocator {

StaticPooling::StaticPooling() {
  pool_id = 0;
  block_count = 0;
  allocted_block_size = 0;
  bytes_in_use = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = DEFAULT_POOL_SIZE;
  prealloc_pool = nullptr;
}

bool StaticPooling::pool_create(synDeviceId deviceID, uint64_t size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  pool_id = deviceID;
  uint64_t free_mem, total_mem;
  synStatus status = synDeviceGetMemoryInfo(deviceID, &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_DEVMEM_DEBUG(
        Logger::formatStatusMsg(status),
        "POOL:: Cannot obtain device memory info.");
  }
  if (size > free_mem) {
    PT_DEVMEM_DEBUG("POOL:: requested size is more than avaiable memory");
    size = 0.9 * free_mem;
    PT_DEVMEM_DEBUG("POOL:: set new pool size : ", size);
  }
  max_pool_size = size;

  auto p = allocateHostMemory(simple_pool_t);
  if (!p) {
    PT_DEVMEM_DEBUG("POOL:: Cannot obtain pool memory");
    return false;
  }

  status = synDeviceMalloc(pool_id, size, 0, 0, &p->memptr);
  if (synStatus::synSuccess != status) {
    freeHostMemory(p);
    PT_DEVMEM_FATAL(
        Logger::formatStatusMsg(status),
        "POOL:: Cannot obtain device memory size.");
    return false;
  }

  p->next = p->memptr;
  p->end = p->next + size;
  p->_start = nullptr;
  p->_top = p->_start;
  PT_DEVMEM_DEBUG("POOL:: simple static pool created");
  pool_allocator::print_device_memory_stats(pool_id);
  prealloc_pool = p;
  stats.pool_id = pool_id;
  stats.memory_limit = max_pool_size;

  print_pool_stats();

  return true;
}

void StaticPooling::pool_destroy() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  simple_pool_t* s_pool = prealloc_pool;

  print_pool_stats();

  if ((s_pool) && (block_count == 0)) {
    if (!pool_allocator::get_device_deallocation()) {
      if (nullptr != (void*)s_pool->memptr) {
        uint64_t ptr_address{reinterpret_cast<uint64_t>(s_pool->memptr)};
        auto status{synDeviceFree(pool_id, ptr_address, 0)};
        if (status) {
          pool_allocator::set_device_deallocation(true);
        }
      }
    }
    pool_allocator::set_device_deallocation(false);

    s_pool->memptr = 0;
    auto chunk = s_pool->_start;
    Poolchunk* chunk_next = s_pool->_start;
    while (chunk != nullptr) {
      chunk_next = chunk->next;
      freeHostMemory(chunk);
      chunk = chunk_next;
    }
    freeHostMemory(s_pool);
    s_pool = nullptr;
    PT_DEVMEM_DEBUG("POOL:: simple static pool destroyed");
  } else {
    PT_DEVMEM_DEBUG("POOL:: cannot destroy pool -- active blocks !!");
    PT_DEVMEM_DEBUG("POOL:: total active blocks :: ", block_count);
  }
  pool_id = 0;
  block_count = 0;
  allocted_block_size = 0;
  bytes_in_use = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = DEFAULT_POOL_SIZE;
  prealloc_pool = nullptr;
}

static uint64_t pool_available(simple_pool_t* p) {
  return p->end - p->next;
}

void StaticPooling::print_pool_stats() const {
  static const std::string occupancy_mask = "[+++]";
  static const std::string free_mask = "[---]";
  static std::stringstream pool_status;
  simple_pool_t* s_pool = prealloc_pool;
  auto chunk = s_pool->_start;
  static int total_blocks = 0;
  while (chunk != nullptr) {
    if (chunk->used) {
      total_blocks++;
      pool_status << occupancy_mask;
    } else if (!chunk->used) {
      pool_status << free_mask;
      free_chunks++;
      free_chunks_size += chunk->size;
    }
    chunk = chunk->next;
  }
  PT_DEVMEM_DEBUG("POOL:: total_blocks in the pool :: ", total_blocks);
  PT_DEVMEM_DEBUG("POOL:: free chunks in the pool :: ", free_chunks);
  PT_DEVMEM_DEBUG("POOL:: free chunks size in the pool :: ", free_chunks_size);
  PT_DEVMEM_DEBUG("POOL::{}", pool_status.str());
  free_chunks = 0;
  free_chunks_size = 0;
  pool_status.str("");
  pool_status.clear();
  return;
}

void* StaticPooling::get_free_chunk(void* ptr, uint64_t size) const {
  auto chunk = (Poolchunk*)ptr;
  // same sized free blocks are reused
  while (chunk != nullptr) {
    if ((chunk->size != size) || (chunk->used)) {
      // PT_DEVMEM_DEBUG("POOL:: size = ", size, " chunk->size = ",
      // chunk->size, " chunk->used = ", chunk->used);
      chunk = chunk->next;
      continue;
    }
    return chunk;
  }
  return nullptr;
}

void* StaticPooling::reuse_chunks(uint64_t size) const {
  simple_pool_t* p = prealloc_pool;
  auto chunk = p->_start;
  auto free_chunk = (Poolchunk*)get_free_chunk(chunk, size);
  if (free_chunk == nullptr) {
    PT_DEVMEM_DEBUG("POOL:: no more reusable chunk: extend pool !!");
    return nullptr;
  }
  PT_DEVMEM_DEBUG("POOL:: reusing preallocated chunk");
  free_chunk->used = true;
  stats.UpdateStats(free_chunk->size, true);
  return (void*)free_chunk->memptr;
}

void* StaticPooling::pool_alloc_chunk(uint64_t size, bool is_workspace) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  simple_pool_t* p = prealloc_pool;
  if (prealloc_pool != p) {
    PT_DEVMEM_FATAL("POOL:: alloc unknown pool !!");
  }

  auto old_chunk = reuse_chunks(size);
  if (old_chunk) {
    ++block_count;
    bytes_in_use += size;
    if (is_workspace)
      stats.scratch_mem_in_use = size;
    log_synDeviceAlloc(reinterpret_cast<uint64_t>(old_chunk), size);
    return old_chunk;
  }
  if (pool_available(p) < size) {
    // TBD: implement better algorithms
    pool_allocator::print_device_memory_stats(pool_id);
    PT_DEVMEM_DEBUG("POOL:: pool exhausted !! deframgment pool ?");
    log_synDeviceAlloc(0, size);
    return nullptr;
  }

  // create a chunk
  auto chunk = allocateHostMemory(Poolchunk);
  if (!chunk) {
    PT_DEVMEM_DEBUG("POOL:: Cannot create a chunk");
    log_synDeviceAlloc(0, size);
    return nullptr;
  }
  chunk->memptr = (uint64_t)p->next;
  chunk->size = size;
  chunk->used = true;
  chunk->next = nullptr;

  if (p->_start == nullptr) {
    p->_start = chunk;
  }
  // Chain the blocks.
  if (p->_top != nullptr) {
    p->_top->next = chunk;
  }
  p->_top = chunk;
  p->next += size;
  allocted_block_size += size;
  ++block_count;
  PT_DEVMEM_DEBUG("POOL:: Allocated block_count :: ", block_count);
  bytes_in_use += size;
  stats.UpdateStats(chunk->size, true, is_workspace);
  log_synDeviceAlloc(chunk->memptr, size);
  return (void*)chunk->memptr;
}

void StaticPooling::pool_free_chunk(void* ptr) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  log_synDeviceDeallocate(reinterpret_cast<uint64_t>(ptr));
  simple_pool_t* s_pool = prealloc_pool;
  PT_DEVMEM_DEBUG("POOL:: freeing block_count :: ", block_count);
  auto chunk = s_pool->_start;
  while (chunk != nullptr) {
    if (chunk->memptr == (uint64_t)ptr) {
      chunk->used = false;
      bytes_in_use -= chunk->size;
      stats.UpdateStats(chunk->size, false);
      break;
    }
    chunk = chunk->next;
  }
  --block_count;
  if (block_count == 0) {
    PT_DEVMEM_DEBUG("POOL:: All blocks freed before pool deletion !");
  }
}

void* StaticPooling::extend_high_memory_allocation(
    uint64_t size,
    [[maybe_unused]] size_t current_ws_size) const {
  PT_DEVMEM_DEBUG(
      "POOL:: Dynamic Pool - extending high memory allocation not supported size::",
      size);
  return nullptr;
}

std::vector<std::pair<uint64_t, uint64_t>> StaticPooling::
    get_occupied_chunk_map() const {
  std::vector<std::pair<uint64_t, uint64_t>> occupied_chunks_map{};
  PT_DEVMEM_WARN("get_occupied_chunk_map not implemented for StaticPooling!");
  return occupied_chunks_map;
}

void StaticPooling::get_stats(MemoryStats* mem_stats) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  *mem_stats = stats;
}

void StaticPooling::clear_stats() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  stats.num_allocs = 0;
  stats.num_frees = 0;
  stats.peak_bytes_in_use = stats.bytes_in_use;
  stats.largest_alloc_size = 0;
}

void StaticPooling::reset_peak_mem_stats() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  stats.peak_bytes_in_use = 0;
}

DynamicPooling::DynamicPooling() {
  pool_id = 0;
  pool_start = nullptr;
  top = pool_start;
  bytes_in_use = 0;
}

void DynamicPooling::freeBlocks(Block* block) const {
  while (block != nullptr) {
    auto next = block->next;
    // std::cerr << "POOL:: synDeviceFree :: block->memptr :: "<<
    // (uint64_t*)block->memptr << std::endl;
    if (!pool_allocator::get_device_deallocation()) {
      if (nullptr != (void*)block->memptr) {
        uint64_t ptr_address{reinterpret_cast<uint64_t>(block->memptr)};
        auto status{synDeviceFree(pool_id, ptr_address, 0)};
        if (status) {
          pool_allocator::set_device_deallocation(true);
        }
      }
    }
    block->memptr = 0;
    freeHostMemory(block);
    block = next;
  }
  pool_allocator::set_device_deallocation(false);
}

void DynamicPooling::freeUnusedBlocks(Block* block) const {
  while (block != nullptr) {
    auto next = block->next;
    if (!block->used && block->memptr) {
      // std::cerr << "POOL:: synDeviceFree :: block->memptr :: "<<
      // (uint64_t*)block->memptr << std::endl;
      if (nullptr != (void*)block->memptr) {
        uint64_t ptr_address{reinterpret_cast<uint64_t>(block->memptr)};
        auto status{synDeviceFree(pool_id, ptr_address, 0)};
        if (status) {
          PT_DEVMEM_DEBUG(
              Logger::formatStatusMsg(status),
              "POOL:: freeUnusedBlocks synDeviceFree failed :: ");
        }
      }
      block->memptr = 0;
      block->size = 0;
      block->used = true;
    }
    block = next;
  }
}

Block* DynamicPooling::retrieveBlock(void* data) const {
  auto block = pool_start;
  while (block != nullptr) {
    if (!block->used ||
        reinterpret_cast<uint64_t*>(block->memptr) !=
            reinterpret_cast<uint64_t*>(data)) {
      block = block->next;
      continue;
    }
    // Found the block:
    return block;
  }
  return nullptr;
}

void DynamicPooling::freeBlock(void* data) const {
  auto block = retrieveBlock(data);
  if (block) {
    block->used = false;
    bytes_in_use -= block->size;
    stats.UpdateStats(block->size, false);
  }
}

Block* DynamicPooling::requestNewBlock(uint64_t size) const {
  // create block header
  auto block = allocateHostMemory(Block);
  if (!block) {
    PT_DEVMEM_DEBUG("POOL:: Cannot create block header");
    return nullptr;
  }

  auto status = synDeviceMalloc(pool_id, size, 0, 0, &block->memptr);
  if (synStatus::synSuccess != status) {
    pool_allocator::print_device_memory_stats(pool_id);
    freeUnusedBlocks(pool_start);
    pool_allocator::print_device_memory_stats(pool_id);
    PT_DEVMEM_DEBUG("POOL:: Reusing freed fragments for size :: ", size);
    auto status = synDeviceMalloc(pool_id, size, 0, 0, &block->memptr);
    if (synStatus::synSuccess != status) {
      freeHostMemory(block);
      PT_DEVMEM_DEBUG(
          Logger::formatStatusMsg(status),
          "POOL:: Cannot obtain device memory size.");
      return nullptr;
    }
  }
  // std::cerr << "POOL:: synDeviceMalloc :: block->memptr :: "<<
  // (uint64_t*)block->memptr << " size :: " << size <<std::endl;
  PT_DEVMEM_DEBUG("POOL:: Creating a new block of size :: ", size);
  return block;
}

Block* DynamicPooling::equalFit(uint64_t size) const {
  auto block = pool_start;
  while (block != nullptr) {
    // same sized free blocks are reused
    if (block->used || block->size != size) {
      block = block->next;
      continue;
    }
    // Found the block:
    PT_DEVMEM_DEBUG(
        "POOL:: Reusing Block of size :: ",
        size,
        "  in block size :: ",
        block->size);
    // ensure block is getting reused
    block->used = true;
    return block;
  }
  return nullptr;
}

Block* DynamicPooling::findBlock(uint64_t size) const {
  return equalFit(size);
}

void* DynamicPooling::allocBlock(uint64_t size) const {
  if (auto block = findBlock(size)) {
    bytes_in_use += block->size;
    stats.UpdateStats(block->size, true);
    return reinterpret_cast<void*>(block->memptr);
  }

  auto block = requestNewBlock(size);
  if (!block) {
    return nullptr;
  }
  block->size = size;
  block->used = true;
  block->next = nullptr;

  // Init Pool.
  if (pool_start == nullptr) {
    pool_start = block;
  }
  // Chain the blocks.
  if (top != nullptr) {
    top->next = block;
  }
  top = block;

  bytes_in_use += block->size;
  stats.UpdateStats(block->size, true);
  return reinterpret_cast<void*>(block->memptr);
}

bool DynamicPooling::pool_create(synDeviceId deviceID, uint64_t size) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  PT_DEVMEM_DEBUG("POOL:: Dynamic Pool Initiated size::", size);
  pool_id = deviceID;
  stats.pool_id = pool_id;
  top = pool_start;
  pool_allocator::print_device_memory_stats(pool_id);
  return true;
}

void DynamicPooling::pool_destroy() const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  freeBlocks(pool_start);
  pool_start = nullptr;
  pool_id = 0;
  top = pool_start;
  bytes_in_use = 0;
  PT_DEVMEM_DEBUG("POOL:: Dynamic Pool destroyed");
  return;
}

void* DynamicPooling::pool_alloc_chunk(uint64_t size, bool is_workspace) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  auto ptr = allocBlock(size);
  if (ptr && is_workspace)
    stats.scratch_mem_in_use = size;
  log_synDeviceAlloc(reinterpret_cast<uint64_t>(ptr), size);
  return ptr;
}

void DynamicPooling::pool_free_chunk(void* ptr) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  log_synDeviceDeallocate(reinterpret_cast<uint64_t>(ptr));
  freeBlock(ptr);
}

void* DynamicPooling::extend_high_memory_allocation(
    uint64_t size,
    [[maybe_unused]] size_t current_ws_size) const {
  PT_DEVMEM_DEBUG(
      "POOL:: Dynamic Pool - extending high memory allocation not supported size::",
      size);
  return nullptr;
}

std::vector<std::pair<uint64_t, uint64_t>> DynamicPooling::
    get_occupied_chunk_map() const {
  std::vector<std::pair<uint64_t, uint64_t>> occupied_chunks_map{};
  PT_DEVMEM_WARN("get_occupied_chunk_map not implemented for DynamicPooling!");
  return occupied_chunks_map;
}

void DynamicPooling::get_stats(MemoryStats* mem_stats) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  *mem_stats = stats;
}

void DynamicPooling::clear_stats() const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  stats.num_allocs = 0;
  stats.num_frees = 0;
  stats.peak_bytes_in_use = stats.bytes_in_use;
  stats.largest_alloc_size = 0;
}

void DynamicPooling::reset_peak_mem_stats() const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  stats.peak_bytes_in_use = 0;
}

} // namespace pool_allocator
} // namespace synapse_helpers
