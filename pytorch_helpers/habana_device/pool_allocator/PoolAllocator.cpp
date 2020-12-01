/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <synapse_api.h>

#include <habana_helpers/logging.h>
#include "../HPUAllocator.h"
#include "../HPUCheck.h"
#include "../HPUGuardImpl.h"
#include "../hpu_cached_devices.h"
#include "PoolAllocator.h"
#include "utils.h"

namespace at {
namespace habana {
namespace pool_allocator {

StaticPooling::StaticPooling() {
  pool_id = 0;
  block_count = 0;
  allocted_block_size = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = DEFAULT_POOL_SIZE;
  prealloc_pool = nullptr;
}

void* StaticPooling::pool_create(synDeviceId deviceID, uint64_t size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  size = pool_allocator::block_align(size);
  pool_id = deviceID;
  uint64_t free_mem, total_mem;
  auto status = synDeviceGetMemoryInfo(deviceID, &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_DEVICE_DEBUG(
        "POOL:: Cannot obtain device memory info. Status: ", status);
  }
  if (size > free_mem) {
    PT_DEVICE_DEBUG("POOL:: requested size is more than avaiable memory");
    size = 0.9 * free_mem;
    PT_DEVICE_DEBUG("POOL:: set new pool size : ", size);
  }
  max_pool_size = size;

  auto p = allocateHostMemory(simple_pool_t);
  if (!p) {
    PT_DEVICE_DEBUG("POOL:: Cannot obtain pool memory");
    return nullptr;
  }

  status = synDeviceMalloc(pool_id, size, 0, 0, &p->memptr);
  if (synStatus::synSuccess != status) {
    freeHostMemory(p);
    PT_DEVICE_FATAL(
        "POOL:: Cannot obtain device memory size. Status: ", status);
    return nullptr;
  }

  p->next = p->memptr;
  p->end = p->next + size;
  p->_start = nullptr;
  p->_top = p->_start;
  PT_DEVICE_DEBUG("POOL:: simple static pool created");
  pool_allocator::print_device_memory_stats(pool_id);
  prealloc_pool = p;

  print_pool_stats();

  return p;
}

void StaticPooling::pool_destroy(void* ptr) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  simple_pool_t* s_pool = prealloc_pool;

  print_pool_stats();

  if ((s_pool) && (block_count == 0)) {
    if (!pool_allocator::get_device_deallocation()) {
      if (nullptr != (void*)s_pool->memptr) {
        // std::cerr << "POOL:: synDeviceFree :: s_pool->memptr :: "<<
        // s_pool->memptr << std::endl;
        uint64_t ptr_address{reinterpret_cast<uint64_t>(s_pool->memptr)};
        auto status{synDeviceFree(pool_id, ptr_address, 0)};
        if (status) {
          // TORCH_HABANA_CHECK(status, "synDeviceFree failed");
          pool_allocator::set_device_deallocation(true);
        }
      }
    }
    pool_allocator::set_device_deallocation(false);

    s_pool->memptr = 0;
    auto chunk = s_pool->_start;
    while (chunk != nullptr) {
      freeHostMemory(chunk);
      chunk = chunk->next;
    }
    freeHostMemory(s_pool);
    s_pool = nullptr;
    PT_DEVICE_DEBUG("POOL:: simple static pool destroyed");
  } else {
    PT_DEVICE_DEBUG("POOL:: cannot destroy pool -- active blocks !!");
    PT_DEVICE_DEBUG("POOL:: total active blocks :: ", block_count);
  }
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
  PT_DEVICE_DEBUG("POOL:: total_blocks in the pool :: ", total_blocks);
  PT_DEVICE_DEBUG("POOL:: free chunks in the pool :: ", free_chunks);
  PT_DEVICE_DEBUG("POOL:: free chunks size in the pool :: ", free_chunks_size);
  PT_DEVICE_DEBUG("POOL::{}", pool_status.str());
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
      // PT_DEVICE_DEBUG("POOL:: size = ", size, " chunk->size = ", chunk->size,
      // " chunk->used = ", chunk->used);
      chunk = chunk->next;
      continue;
    }
    return chunk;
  }
  return nullptr;
}

void* StaticPooling::reuse_chunks(void* ptr, uint64_t size) const {
  simple_pool_t* p = (simple_pool_t*)ptr;
  auto chunk = p->_start;
  auto free_chunk = (Poolchunk*)get_free_chunk(chunk, size);
  if (free_chunk == nullptr) {
    PT_DEVICE_DEBUG("POOL:: no more reusable chunk: extend pool !!");
    // print_pool_stats();
    return nullptr;
  }
  PT_DEVICE_DEBUG("POOL:: reusing preallocated chunk");
  free_chunk->used = true;
  return (void*)free_chunk->memptr;
}

void* StaticPooling::pool_alloc_chunk(void* ptr, uint64_t size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  size = pool_allocator::block_align(size);
  simple_pool_t* p = (simple_pool_t*)ptr;
  if (prealloc_pool != p) {
    PT_DEVICE_FATAL("POOL:: alloc unknown pool !!");
  }

  auto old_chunk = reuse_chunks(ptr, size);
  if (old_chunk) {
    ++block_count;
    return old_chunk;
  }

  if (pool_available(p) < size) {
    // TBD: implement better algorithms
    pool_allocator::print_device_memory_stats(pool_id);
    print_pool_stats();
    PT_DEVICE_DEBUG("POOL:: pool exhausted !! deframgment pool ?");
    return nullptr;
  }

  // create a chunk
  auto chunk = allocateHostMemory(Poolchunk);
  if (!chunk) {
    PT_DEVICE_DEBUG("POOL:: Cannot create a chunk");
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
  PT_DEVICE_DEBUG("POOL:: Allocated block_count :: ", block_count);
  return (void*)chunk->memptr;
}

void StaticPooling::pool_free_chunk(void* ptr) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  simple_pool_t* s_pool = prealloc_pool;
  PT_DEVICE_DEBUG("POOL:: freeing block_count :: ", block_count);
  auto chunk = s_pool->_start;
  while (chunk != nullptr) {
    if (chunk->memptr == (uint64_t)ptr) {
      chunk->used = false;
      break;
    }
    chunk = chunk->next;
  }
  --block_count;
  if (block_count == 0) {
    print_pool_stats();
    PT_DEVICE_DEBUG("POOL:: All blocks freed before pool deletion !");
  }
}

DynamicPooling::DynamicPooling() {
  pool_id = 0;
  pool_start = nullptr;
  top = pool_start;
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
        // TORCH_HABANA_CHECK(status, "synDeviceFree failed");
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
          PT_DEVICE_DEBUG(
              "POOL:: freeUnusedBlocks synDeviceFree failed :: ", status);
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
  }
}

Block* DynamicPooling::requestNewBlock(uint64_t size) const {
  // create block header
  auto block = allocateHostMemory(Block);
  if (!block) {
    PT_DEVICE_DEBUG("POOL:: Cannot create block header");
    return nullptr;
  }

  auto status = synDeviceMalloc(pool_id, size, 0, 0, &block->memptr);
  if (synStatus::synSuccess != status) {
    pool_allocator::print_device_memory_stats(pool_id);
    freeUnusedBlocks(pool_start);
    pool_allocator::print_device_memory_stats(pool_id);
    PT_DEVICE_DEBUG("POOL:: Reusing freed fragments for size :: ", size);
    auto status = synDeviceMalloc(pool_id, size, 0, 0, &block->memptr);
    if (synStatus::synSuccess != status) {
      freeHostMemory(block);
      PT_DEVICE_DEBUG(
          "POOL:: Cannot obtain device memory size. Status: ", status);
      return nullptr;
    }
  }
  // std::cerr << "POOL:: synDeviceMalloc :: block->memptr :: "<<
  // (uint64_t*)block->memptr << " size :: " << size <<std::endl;
  PT_DEVICE_DEBUG("POOL:: Creating a new block of size :: ", size);
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
    PT_DEVICE_DEBUG(
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

  return reinterpret_cast<void*>(block->memptr);
}

void* DynamicPooling::pool_create(synDeviceId deviceID, uint64_t size) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  PT_DEVICE_DEBUG("POOL:: Dynamic Pool Initiated");
  size = pool_allocator::block_align(size);
  pool_id = deviceID;
  pool_allocator::print_device_memory_stats(pool_id);
  return pool_start;
}

void DynamicPooling::pool_destroy(void* ptr) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  freeBlocks(pool_start);
  pool_start = nullptr;
  PT_DEVICE_DEBUG("POOL:: Dynamic Pool destroyed");
  return;
}

void* DynamicPooling::pool_alloc_chunk(void* ptr, uint64_t size) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  size = pool_allocator::block_align(size);
  return allocBlock(size);
}

void DynamicPooling::pool_free_chunk(void* ptr) const {
  const std::lock_guard<std::mutex> lock(vp_mutex);
  freeBlock(ptr);
}

} // namespace pool_allocator
} // namespace habana
} // namespace at
