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
#include <synapse_api.h>

#include <habana_helpers/logging.h>
#include "CoalescedStringentPoolAllocator.h"
#include "backend/synapse_helpers/devmem_logger.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/lightweight_memory_usage_logger.h"
#include "backend/synapse_helpers/util.h"

#define DEFRAGMENT_TH(arg) std::ceil(0.9 * (arg))

namespace synapse_helpers {
namespace pool_allocator {

Bin* BinUtils::BinFromIndex(uint64_t index) const {
  Bin* bin = const_cast<Bin*>(
      reinterpret_cast<const Bin*>(&(bins_space[index * sizeof(Bin)])));
  return bin;
}

uint64_t BinUtils::BinIndexForSize(size_t bytes) const {
  uint64_t v =
      std::max<size_t>(bytes, kMinAllocationSize) >> kMinAllocationBits;
  uint64_t index = std::min(kNumBins - 1, Log2FloorNonZero(v));
  return index;
}

Bin* BinUtils::BinForSize(size_t bytes) const {
  return BinFromIndex(BinIndexForSize(bytes));
}

void BinUtils::InsertFreeChunkIntoBin(Chunk* c) const {
  if (!c->used && (c->bin_index == kInvalidBinNum)) {
    uint64_t bin_index = BinIndexForSize(c->size);
    Bin* new_bin = BinFromIndex(bin_index);
    c->bin_index = bin_index;
    new_bin->free_chunks.insert(c);
  } else {
    PT_DEVMEM_DEBUG(
        "CS_POOL:: InsertFreeChunkIntoBin - memptr = ",
        uint64_to_hex_string(c->memptr),
        ", bin_index = ",
        c->bin_index);
    PT_DEVMEM_DEBUG(
        "incorrect chunk in use", c->used, "with bin index", c->bin_index);
  }
}

void BinUtils::RemoveFreeChunkFromBin(Chunk* c) const {
  if (!c->used && (c->bin_index != kInvalidBinNum)) {
    int count = BinFromIndex(c->bin_index)->free_chunks.erase(c);
    if (count < 0) {
      PT_DEVMEM_DEBUG(
          "CS_POOL:: RemoveFreeChunkFromBin - memptr = ",
          uint64_to_hex_string(c->memptr),
          ", bin_index = ",
          c->bin_index);
      PT_DEVMEM_DEBUG("could not find chunk in bin");
    } else {
      c->bin_index = kInvalidBinNum;
    }
  } else {
    PT_DEVMEM_DEBUG(
        "CS_POOL:: RemoveFreeChunkFromBin - memptr = ",
        uint64_to_hex_string(c->memptr),
        ", bin_index = ",
        c->bin_index);
    PT_DEVMEM_DEBUG(
        "incorrect chunk in use", c->used, "with bin index", c->bin_index);
  }
}

void BinUtils::RemoveFreeChunkIterFromBin(
    Bin::FreeChunkSet* free_chunks,
    const Bin::FreeChunkSet::iterator& citer) const {
  Chunk* c = *citer;
  HABANA_ASSERT(!c->used && (c->bin_index != kInvalidBinNum));
  free_chunks->erase(citer);
  c->bin_index = kInvalidBinNum;
}

CoalescedStringentPooling::CoalescedStringentPooling(device& device)
    : device_{device} {
  pool_id = 0;
  chunk_count = 0;
  allocted_chunk_size = 0;
  bytes_in_use = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = default_pool_size;
  prealloc_pool = nullptr;
  small_allocs_ = nullptr;
  defragmenter_state_started_ = 0;
  alignment = device_.get_device_memory_alignment();
  kMinAllocationBits = std::log2(alignment) + 1;
  kMinAllocationSize = 1 << kMinAllocationBits;
  bin_utils = new BinUtils(kMinAllocationSize, kMinAllocationBits);
  header_bytes = alignment;
}

CoalescedStringentPooling::~CoalescedStringentPooling() {
  realtime_logger_.reset();
  if (small_allocs_) {
    small_allocs_->Reset();
    small_allocs_ = nullptr;
  }
  delete bin_utils;
  bin_utils = nullptr;
}

void CoalescedStringentPooling::set_defragmenter_state(bool started) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  defragmenter_state_started_ = started;
}

bool CoalescedStringentPooling::pool_create(synDeviceId deviceID, uint64_t size)
    const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  synStatus status{synStatus::synSuccess};
  pool_id = deviceID;
  uint64_t free_mem, total_mem;
  status = synDeviceGetMemoryInfo(deviceID, &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_DEVMEM_DEBUG(
        Logger::formatStatusMsg(status),
        "CS_POOL:: Cannot obtain device memory info.");
  }

  // try to take max free memory when not set by user
  if ((size > free_mem) || (size == default_pool_size)) {
    // leave small factor of memory for synapse to use, so set acquire 100% of
    // free memory
    // Some memory needs to be left for intermediate buffer for collective
    // inside HCCL.
    std::size_t hccl_allowance_bytes = 0;
    const char* world_size_s = std::getenv("WORLD_SIZE");
    if (world_size_s != nullptr && atoi(world_size_s) > 1) {
      const std::size_t HCCL_MEMORY_ALLOWANCE_MB{
          GET_ENV_FLAG_NEW(PT_HCCL_MEMORY_ALLOWANCE_MB)};
      hccl_allowance_bytes = 1048576 * HCCL_MEMORY_ALLOWANCE_MB;
    }
    stats.pre_allocate_size += hccl_allowance_bytes;

    auto val = GET_ENV_FLAG_NEW(PT_HPU_POOL_MEM_ACQUIRE_PERC);
    uint32_t mem_acquire_perc = (val > 100) ? 100 : val;
    size = ((mem_acquire_perc / 100.0) * free_mem) - hccl_allowance_bytes;

    PT_DEVMEM_DEBUG(
        "CS_POOL:: use 100% of freepool size, free mem :: ",
        free_mem,
        "hccl allowance bytes",
        hccl_allowance_bytes,
        " size used for pool :: ",
        size);
  }
  max_pool_size = size;

  auto p = new simple_coalesced_pool_t();
  if (!p) {
    PT_DEVMEM_DEBUG("CS_POOL:: Cannot obtain pool memory");
    return false;
  }

  status = synDeviceMalloc(pool_id, size, 0, 0, &p->basememptr);
  if (synStatus::synSuccess != status) {
    delete (p);
    PT_DEVMEM_FATAL("CS_POOL:: Cannot obtain device memory size.");
    return false;
  }
  log_synDevicePoolCreate(
      free_mem, GET_ENV_FLAG_NEW(PT_HPU_POOL_MEM_ACQUIRE_PERC), p->basememptr);

  // alignemnt bytes left for future use - header maintence in device memory
  // instead of host
  p->memptr = p->basememptr + header_bytes;
  p->next = p->memptr;
  p->end = p->basememptr + size;
  p->start = nullptr;
  p->top = p->start;
  PT_DEVMEM_DEBUG("CS_POOL:: static coalesced stringent pool created");
  print_device_memory_stats(pool_id);
  prealloc_pool = p;

  PT_DEVMEM_DEBUG(
      "CS_POOL:: Pool Created :: base host :: ",
      p,
      " base ptr :: ",
      uint64_to_hex_string(p->basememptr),
      " memptr :: ",
      uint64_to_hex_string(p->memptr),
      " prealloc_pool :: ",
      uint64_to_hex_string(prealloc_pool->memptr),
      " end :: ",
      uint64_to_hex_string(p->end),
      " next :: ",
      uint64_to_hex_string(p->next));

  log_DRAM_start(p->memptr);
  log_DRAM_size(max_pool_size);
  // We create bins to fit all possible ranges that cover the
  // max_pool_size starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  for (uint64_t b = 0; b < kNumBins; b++) {
    size_t bin_size = bin_utils->BinNumToSize(b);
    PT_DEVMEM_DEBUG("Creating bin of max chunk size ", bin_size);
    new (bin_utils->BinFromIndex(b)) Bin(bin_size);
    HABANA_ASSERT(
        bin_utils->BinForSize(bin_size) == bin_utils->BinFromIndex(b));
    HABANA_ASSERT(
        bin_utils->BinForSize(bin_size + 255) == bin_utils->BinFromIndex(b));
    HABANA_ASSERT(
        bin_utils->BinForSize(bin_size * 2 - 1) == bin_utils->BinFromIndex(b));
    if (b + 1 < kNumBins) {
      HABANA_ASSERT(
          bin_utils->BinForSize(bin_size * 2) != bin_utils->BinFromIndex(b));
    }
  }
  // Create one large chunk for the whole memory space that will be chunked
  // and use later
  Chunk* chunk = new Chunk();
  chunk->memptr = (uint64_t)p->next;
  chunk->extra_space = 0;
  chunk->size = max_pool_size - header_bytes;
  chunk->used = false;
  chunk->next = nullptr;
  chunk->prev = nullptr;

  if (p->start == nullptr) {
    p->start = chunk;
  }
  // Chain the chunks.
  if (p->top != nullptr) {
    chunk->prev = p->top;
    p->top->next = chunk;
  }
  p->top = chunk;
  p->next += size;

  chunks[chunk->memptr] = chunk;
  bin_utils->InsertFreeChunkIntoBin(chunk);

  stats.pool_id = pool_id;
  stats.memory_limit = max_pool_size;
  stats.bytes_in_use += header_bytes;
  bytes_in_use += header_bytes;
  stats.pre_allocate_size += header_bytes;
  size_t small_alloc_size = 6 * 1024 * alignment;
  const auto chunk_ptr = static_cast<int8_t*>(alloc_chunk(
      small_alloc_size, 0 /*default stream*/, false /*use_stream*/));
  const auto free_chunk = [this](int8_t* ptr) { delete_chunk(ptr); };

  small_allocs_ = std::make_unique<SmallAllocs>(
      std::unique_ptr<int8_t, std::function<void(int8_t*)>>(
          chunk_ptr, free_chunk),
      alignment,
      bin_utils->BinNumToSize(0),
      small_alloc_size);
  stats.num_allocs = 0;
  stats.bytes_in_use += small_alloc_size;
  bytes_in_use += small_alloc_size;

  static bool is_realtime_logger_enable =
      GET_ENV_FLAG_NEW(PT_ENABLE_REALTIME_MEMORY_LOGGING);
  if (is_realtime_logger_enable) {
    realtime_logger_ =
        std::make_unique<realtime_logger::RealTimeMeoryLogger>(this);
  }
  MEMORY_MONITORING_SET_DEVICE(this);
  return true;
}

void CoalescedStringentPooling::pool_destroy() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  MEMORY_MONITORING_RESET_DEVICE;

  PT_DEVMEM_DEBUG("CS_POOL:: pool_destroy");
  simple_coalesced_pool_t* s_pool = prealloc_pool;
  {
    std::string updated_msg = "pool_destroy: ";
    updated_msg = updated_msg + "\n" + stats.DebugString();
    PT_DEVMEM_DEBUG("CS_POOL:: pool_destroy stats:", updated_msg.c_str());
  }
  CoalescedStringentPooling::print_pool_stats();

  if ((s_pool) && (chunk_count != 0)) {
    PT_DEVMEM_DEBUG("CS_POOL:: warning -- active chunks !!");
    PT_DEVMEM_DEBUG("CS_POOL:: total active chunks :: ", chunk_count);
  }

  if (s_pool) {
    if (small_allocs_) {
      small_allocs_->Reset();
      small_allocs_ = nullptr;
    }

    if (!get_device_deallocation()) {
      if (nullptr != (void*)s_pool->basememptr) {
        uint64_t ptr_address{reinterpret_cast<uint64_t>(s_pool->basememptr)};
        auto status{synDeviceFree(pool_id, ptr_address, 0)};
        if (status) {
          set_device_deallocation(true);
        }
      }
    }
    set_device_deallocation(false);

    for (uint64_t b = 0; b < kNumBins; b++) {
      Bin* bin = bin_utils->BinFromIndex(b);
      if (bin)
        bin->~Bin();
    }

    s_pool->basememptr = 0;

    for (auto& m : chunks) {
      delete (m.second);
    }
    chunks.clear();
    delete (s_pool);
    s_pool = nullptr;
    PT_DEVMEM_DEBUG("CS_POOL:: static coalesced pool destroyed");
  }
  pool_id = 0;
  chunk_count = 0;
  allocted_chunk_size = 0;
  bytes_in_use = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = default_pool_size;
  high_memory_allocated_ = false;
}

bool CoalescedStringentPooling::is_memory_available(size_t size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);

  if ((size + bytes_in_use) > max_pool_size) {
    PT_DEVMEM_DEBUG("total requested memory size::", size, " not available");
    return false;
  }
  return true;
}

bool CoalescedStringentPooling::is_memory_available(
    size_t persistant_size,
    size_t curr_ws_size,
    size_t new_ws_size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);

  if ((persistant_size + new_ws_size) + (bytes_in_use - curr_ws_size) >
      max_pool_size) {
    PT_DEVMEM_DEBUG(
        "total requested memory size::",
        persistant_size,
        " with new workspace size::",
        new_ws_size,
        " not available");
    return false;
  }
  return true;
}

// Returns a pointer to an underlying allocated chunk of size 'num_bytes'.
void* CoalescedStringentPooling::FindChunkPtr(
    uint64_t bin_index,
    size_t num_bytes,
    hpuStream_t stream,
    bool use_stream) const {
  // First identify the first bin that could satisfy num_bytes.
  for (; bin_index < kNumBins; bin_index++) {
    // Start searching from the first bin for the smallest chunk that fits
    // num_bytes.
    Bin* b = bin_utils->BinFromIndex(bin_index);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      Chunk* chunk = *citer;
      HABANA_ASSERT(!chunk->used);
      bool got_chunk = false;
      if (use_stream) {
        got_chunk =
            (chunk->size >= num_bytes &&
             (chunk->associated_to_stream && stream == chunk->stream));
      } else {
        got_chunk = (chunk->size >= num_bytes);
      }
      if (got_chunk) {
        // We found an existing chunk that fits us that wasn't in use, so remove
        // it from the free bin structure prior to using.
        bin_utils->RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably large
        // pieces, do so.  In any case don't waste more than
        // kMaxInternalFragmentation bytes on padding this alloc.
        const int64_t kMaxInternalFragmentation = 128 << 20; // 128mb
        if ((chunk->size > num_bytes) &&
            (chunk->size >= num_bytes * 2 ||
             static_cast<int64_t>(chunk->size) - num_bytes >=
                 kMaxInternalFragmentation ||
             (num_bytes < DEFRAGMENT_TH(chunk->size)) ||
             defragmenter_state_started_)) {
          try_splitting_chunks(chunk, num_bytes);
        }
        bin_utils->InsertFreeChunkIntoBin(chunk);

        return (void*)(chunk);
      }
    }
  }

  return nullptr;
}

void CoalescedStringentPooling::print_pool_stats() const {
  const std::string occupancy_mask = "[+++]";
  const std::string free_mask = "[00000]";
  std::stringstream pool_status;
  int occupied_chunks = 0;
  int total_chunks = 0;
  uint64_t occupied_size = 0;
  uint64_t total_size = 0;
  int total_extra_spaced_chunks = 0;
  uint64_t total_exta_size = 0;
  uint64_t cntgs_free_chunks_size = 0;
  uint64_t max_cntgs_free_chunks_size = 0;
  pool_status.str("");
  pool_status.clear();

  std::map<uint64_t, Chunk*> chunks_ordered;
  for (auto& m : chunks) {
    chunks_ordered.insert(m);
  }
  for (auto& m : chunks_ordered) {
    auto chunk = m.second;
    total_chunks++;
    total_size += chunk->size;
    if (chunk->extra_space) {
      total_extra_spaced_chunks++;
      total_exta_size += chunk->extra_space;
    }
    if (chunk->used) {
      occupied_chunks++;
      occupied_size += chunk->size;
      pool_status << occupancy_mask;
    } else if (!chunk->used && (chunk->size != 0)) {
      pool_status << free_mask;
      free_chunks++;
      free_chunks_size += chunk->size;
      cntgs_free_chunks_size = getContigousChunkSize(chunk);
      if (max_cntgs_free_chunks_size < cntgs_free_chunks_size) {
        max_cntgs_free_chunks_size = cntgs_free_chunks_size;
      }
      cntgs_free_chunks_size = 0;

      if (chunk->prev && !chunk->prev->used && chunk->prev->size) {
        PT_DEVMEM_DEBUG(
            "CS_POOL:: can be merged :: chunk :: ",
            uint64_to_hex_string(chunk->memptr),
            " with prev :: ",
            uint64_to_hex_string(chunk->prev->memptr));
      }
      if (chunk->next && !chunk->next->used && chunk->next->size) {
        PT_DEVMEM_DEBUG(
            "CS_POOL:: can be merged :: chunk :: ",
            uint64_to_hex_string(chunk->memptr),
            " with next :: ",
            uint64_to_hex_string(chunk->next->memptr));
      }
    }
  }
  PT_DEVMEM_DEBUG("CS_POOL:: total_chunks in the pool :: ", total_chunks);
  PT_DEVMEM_DEBUG("CS_POOL:: total_size in the pool :: ", total_size);
  PT_DEVMEM_DEBUG("CS_POOL:: occupied_chunks in the pool :: ", occupied_chunks);
  PT_DEVMEM_DEBUG(
      "CS_POOL:: occupied_chunks size in the pool :: ", occupied_size);
  PT_DEVMEM_DEBUG("CS_POOL:: free chunks in the pool :: ", free_chunks);
  PT_DEVMEM_DEBUG(
      "CS_POOL:: free chunks size in the pool :: ", free_chunks_size);
  PT_DEVMEM_DEBUG(
      "CS_POOL:: max contigous chunks size in the pool :: ",
      max_cntgs_free_chunks_size);
  PT_DEVMEM_DEBUG(
      "CS_POOL:: total_extra_spaced_chunks in the pool  :: ",
      total_extra_spaced_chunks);
  PT_DEVMEM_DEBUG(
      "CS_POOL:: total_extra_size in the pool chunks :: ", total_exta_size);
  PT_DEVMEM_DEBUG("CS_POOL::{}", pool_status.str());
  PT_DEVMEM_DEBUG(
      "CS_POOL::Fragmentation = ",
      1 - ((double)max_cntgs_free_chunks_size / free_chunks_size));
  total_chunks = 0;
  total_size = 0;
  occupied_chunks = 0;
  occupied_size = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  pool_status.str("");
  pool_status.clear();
  return;
}

size_t CoalescedStringentPooling::get_max_cntgs_chunk_size() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);

  uint64_t max_cntgs_free_chunks_size = 0;
  uint64_t cntgs_free_chunks_size = 0;

  std::map<uint64_t, Chunk*> chunks_ordered;
  for (auto& m : chunks) {
    chunks_ordered.insert(m);
  }
  for (auto& m : chunks_ordered) {
    auto chunk = m.second;

    if (!chunk->used && (chunk->size != 0)) {
      cntgs_free_chunks_size = getContigousChunkSize(chunk);
      if (max_cntgs_free_chunks_size < cntgs_free_chunks_size) {
        max_cntgs_free_chunks_size = cntgs_free_chunks_size;
      }
      cntgs_free_chunks_size = 0;
    }
  }

  return max_cntgs_free_chunks_size;
}

bool CoalescedStringentPooling::isChunkContigous(Chunk* chunk1, Chunk* chunk2)
    const {
  if ((chunk1->memptr + chunk1->size) == chunk2->memptr) {
    return true;
  }
  return false;
}

uint64_t CoalescedStringentPooling::getContigousChunkSize(Chunk* chunk) const {
  uint64_t ctgs_chunks_size = 0;
  uint64_t ctgs_chunks = 0;
  auto temp1 = chunk;
  while (temp1 && temp1->prev && !temp1->prev->used &&
         isChunkContigous(temp1->prev, temp1)) {
    ctgs_chunks++;
    ctgs_chunks_size += temp1->prev->size;
    temp1 = temp1->prev;
  };
  auto temp2 = chunk;
  while (temp2 && temp2->next && !temp2->next->used &&
         isChunkContigous(temp2, temp2->next)) {
    ctgs_chunks++;
    ctgs_chunks_size += temp2->next->size;
    temp2 = temp2->next;
  };
  ctgs_chunks_size += chunk->size;

  return ctgs_chunks_size;
}

Chunk* CoalescedStringentPooling::reuse_chunks(
    uint64_t size,
    hpuStream_t stream,
    bool use_stream) const {
  int bin_index = bin_utils->BinIndexForSize(size);
  Chunk* free_chunk = (Chunk*)FindChunkPtr(bin_index, size, stream, use_stream);
  if (free_chunk == nullptr) {
    PT_DEVMEM_DEBUG(
        "CS_POOL:: no more reusable chunk: defragment or extend !!");
    return nullptr;
  }
  bin_utils->RemoveFreeChunkFromBin(free_chunk);
  free_chunk->used = true;
  free_chunk->associated_to_stream = true;
  free_chunk->stream = stream;
  return free_chunk;
}

void* CoalescedStringentPooling::extend_high_memory_allocation(
    uint64_t size,
    size_t current_ws_size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  if (size > max_pool_size) {
    PT_DEVMEM_DEBUG("CS_POOL:: alloc size exceeds max size !!");
    log_synDeviceWorkspace(0, size);
    return nullptr;
  }

  // get tail chunk
  if (prealloc_pool == nullptr) {
    PT_DEVMEM_FATAL("CS_POOL:: alloc invalid pool !!");
  }
  Chunk* tail_chunk = prealloc_pool->top;

  if (tail_chunk == nullptr) {
    PT_DEVMEM_FATAL("CS_POOL:: extend_high_memory_alloc tail chunk invalid!!");
  }

  // high memory is not allocated and tail chunk used
  if (!high_memory_allocated_ && tail_chunk->used) {
    PT_DEVMEM_DEBUG(
        "CS_POOL:: no space for high memory allocation, already allocated!!");
    log_synDeviceWorkspace(0, size);
    return nullptr;
  }

  // if high_memory is already allocated, check the size with the requested size
  if (high_memory_allocated_ && (size <= tail_chunk->size) &&
      tail_chunk->used) {
    log_synDeviceWorkspace(tail_chunk->memptr, size);
    return (void*)tail_chunk->memptr;
  }

  // if high memory is already allocated, extend the remaining memory for the
  // requested size
  if (high_memory_allocated_) {
    tail_chunk->used = false;
    bin_utils->InsertFreeChunkIntoBin(try_to_merge(tail_chunk));
    tail_chunk = prealloc_pool->top;
  }

  if (tail_chunk->size < size) {
    PT_DEVMEM_DEBUG(
        "CS_POOL:: out of memory, when trying to extend high meory for size::",
        size);
    bin_utils->RemoveFreeChunkFromBin(tail_chunk);
    tail_chunk->used = true;
    log_synDeviceWorkspace(0, size);
    return nullptr;
  }

  // mark high_memory allocated to true and check if we need to split
  high_memory_allocated_ = true;
  auto size_left = tail_chunk->size - size;

  if (tail_chunk->bin_index == kInvalidBinNum) {
    PT_DEVMEM_FATAL("CS_POOL:: extend_high_memory_alloc tail chunk invalid!!");
  }

  if (size_left > 0) {
    Bin* bin = bin_utils->BinFromIndex(tail_chunk->bin_index);
    auto tail_chunk_itr = bin->free_chunks.find(tail_chunk);
    bin_utils->RemoveFreeChunkIterFromBin(&bin->free_chunks, tail_chunk_itr);

    try_splitting_chunks(tail_chunk, size_left);
    bin_utils->InsertFreeChunkIntoBin(tail_chunk);
    // split will return the size_left chunk, the actual requested chunk wil be
    // split_chunk->next.
    tail_chunk = tail_chunk->next;
  }
  bin_utils->RemoveFreeChunkFromBin(tail_chunk);
  tail_chunk->used = true;
  PT_DEVMEM_DEBUG(
      "Update status for new workspace allocation:: extra space::",
      (tail_chunk->size - current_ws_size))
  stats.UpdateStats((tail_chunk->size - current_ws_size), true);
  bytes_in_use += (tail_chunk->size - current_ws_size);
  stats.scratch_mem_in_use = tail_chunk->size;
  log_synDeviceWorkspace(tail_chunk->memptr, size);
  return (void*)tail_chunk->memptr;
}

void* CoalescedStringentPooling::pool_alloc_chunk(
    uint64_t size,
    [[maybe_unused]] bool is_workspace) const {
  std::unique_lock<std::mutex> lock(sp_mutex);
  process_events();
  void* ptr = alloc_chunk(size, 0 /*default_stream*/, false /*use_stream*/);
  log_synDeviceAlloc(reinterpret_cast<uint64_t>(ptr), size);
  return ptr;
}

void* CoalescedStringentPooling::pool_alloc_chunk(
    uint64_t size,
    hpuStream_t stream,
    bool use_stream) const {
  std::unique_lock<std::mutex> lock(sp_mutex);
  process_events();
  void* ptr = alloc_chunk(size, stream, use_stream);

  log_synDeviceAlloc(reinterpret_cast<uint64_t>(ptr), size);
  return ptr;
}

void* CoalescedStringentPooling::alloc_chunk(
    uint64_t size,
    hpuStream_t stream,
    bool use_stream) const {
  void* ptr = nullptr;

  if (size % alignment != 0) {
    PT_DEVMEM_FATAL(
        "CS_POOL:: alloc_chunk requested size not aligned to default size::",
        size);
  }

  // now use smallalloc only for default stream
  if (small_allocs_ && stream == 0) {
    ptr = small_allocs_->Allocate(size);
  }

  if (ptr != nullptr) {
    ++stats.num_allocs;
    ++stats.total_allocs;
    return ptr;
  }

  if (size > max_pool_size) {
    PT_DEVMEM_DEBUG("CS_POOL:: alloc size exceeds max size !!");
    return nullptr;
  }

  simple_coalesced_pool_t* p = (simple_coalesced_pool_t*)prealloc_pool;
  if (prealloc_pool != p) {
    PT_DEVMEM_FATAL("CS_POOL:: alloc unknown pool !!");
  }

  auto old_chunk = reuse_chunks(size, stream, use_stream);
  if (old_chunk) {
    ++chunk_count;
    // extra space available in blocks after split/coalasce
    old_chunk->extra_space = old_chunk->size - size;
    bytes_in_use += old_chunk->size;
    stats.UpdateStats(old_chunk->size, true);
    return (void*)old_chunk->memptr;
  }

  PT_DEVMEM_DEBUG("CS_POOL:: pool exhausted !! for size :: ", size);
  return nullptr;
}

void CoalescedStringentPooling::try_splitting_chunks(
    Chunk* chunk,
    uint64_t size) const {
  // Allocate the new chunk
  Chunk* new_chunk = new Chunk();
  // split extracts requested size from the beginning of the given chunk

  HABANA_ASSERT(!chunk->used && (chunk->bin_index == kInvalidBinNum));

  // new chunk starts size after chunk
  new_chunk->memptr = (chunk->memptr) + size;

  // Set the new sizes of the chunks.
  new_chunk->size = chunk->size - size;
  chunk->size = size;

  new_chunk->used = false;
  new_chunk->freed_counter = 0;
  new_chunk->stream = chunk->stream;
  new_chunk->associated_to_stream = chunk->associated_to_stream;

  // maintain the prev and next pointers
  // c1<->c2 ==> c1<->new_chunk<->c2
  Chunk* next = chunk->next;
  new_chunk->prev = chunk;
  new_chunk->next = next;
  chunk->next = new_chunk;
  if (next) {
    next->prev = new_chunk;
  }

  if (prealloc_pool->top == chunk) {
    // update top
    prealloc_pool->top = new_chunk;
  }

  chunks[new_chunk->memptr] = new_chunk;
  // Add the newly free chunk to the free bin.
  bin_utils->InsertFreeChunkIntoBin(new_chunk);
}

void CoalescedStringentPooling::merge(Chunk* c1, Chunk* c2) const {
  if (c2->event_count > 0 || !c2->stream_uses.empty())
    return;

  if (c1->used || c2->used) {
    PT_DEVMEM_FATAL(" Chunk is in use, cannot merge");
  }

  if (c2->prev != c1) {
    PT_DEVMEM_FATAL(
        "Invalid c2 prev pointer prev->",
        c2->prev->memptr,
        " not equal to c1::",
        c1->memptr);
  }
  // check if c1 and c2 address are contigous(addtional check)
  if ((c1->memptr + c1->size) != c2->memptr) {
    PT_DEVMEM_FATAL(
        "c1 & c2 are not contigous c1->memptr:: ",
        uint64_to_hex_string(c1->memptr),
        " c2-?memptr:",
        uint64_to_hex_string(c2->memptr));
  }

  if (prealloc_pool->top == c2) {
    // update top
    prealloc_pool->top = c1;
  }

  // maint the prev & next pointers
  // c1 previous will remain the same, merge c1 ->c2
  // and change the next pointers
  // c1<->c2<->c3 <=merge=> c1<->c3
  Chunk* c3 = c2->next;
  c1->next = c3;

  if (c3)
    c3->prev = c1;

  c1->size += c2->size;
  c1->freed_counter = std::max(c1->freed_counter, c2->freed_counter);

  // Delete the c2 chunks
  /* remove c2 from chunks map*/
  auto it1 = chunks.find(c2->memptr);
  if (it1 != chunks.end()) {
    chunks.erase(it1);
  }

  // Delete c2
  c2->used = false;
  c2->extra_space = 0;
  c2->size = 0;
  c2->memptr = 0;
  c2->next = nullptr;
  c2->prev = nullptr;

  delete c2;
}

Chunk* CoalescedStringentPooling::try_to_merge(Chunk* c) const {
  Chunk* coalesced_chunk = c;

  // If the next chunk is free, merge it into c and delete it.
  if (c->next != nullptr && !(c->next)->used) {
    bin_utils->RemoveFreeChunkFromBin(c->next);
    merge(c, c->next);
  }

  // If the previous chunk is free, merge c into it and delete c.
  if (c->prev != nullptr && !(c->prev)->used) {
    coalesced_chunk = c->prev;
    bin_utils->RemoveFreeChunkFromBin(c->prev);
    merge(c->prev, c);
  }

  return coalesced_chunk;
}

void CoalescedStringentPooling::pool_free_chunk(void* ptr) const {
  std::unique_lock<std::mutex> lock(sp_mutex);
  log_synDeviceDeallocate(reinterpret_cast<uint64_t>(ptr));
  if ((uint64_t)ptr == 0) {
    PT_DEVMEM_DEBUG("CS_POOL:: pool_free_chunk null ptr");
    return;
  }
  if (small_allocs_ && small_allocs_->IsAllocated(ptr)) {
    delete_chunk(ptr);
  } else {
    auto it = chunks.find((uint64_t)ptr);
    HABANA_ASSERT(it != chunks.end());
    Chunk* chunk = it->second;
    if (!chunk->stream_uses.empty()) {
      insert_events(chunk);
    } else {
      delete_chunk(ptr);
    }
  }
}

void CoalescedStringentPooling::delete_chunk(void* ptr) const {
  if ((uint64_t)ptr == 0) {
    PT_DEVMEM_DEBUG("CS_POOL:: delete chunk null ptr");
    return;
  }

  if (small_allocs_ && small_allocs_->IsAllocated(ptr)) {
    small_allocs_->Deallocate(ptr);
    ++stats.num_frees;
    ++stats.total_frees;
    return;
  }

  auto it = chunks.find((uint64_t)ptr);
  HABANA_ASSERT(it != chunks.end());
  Chunk* chunk = it->second;
  HABANA_ASSERT(chunk->used);
  chunk->used = false;
  chunk->bin_index = kInvalidBinNum;
  chunk->extra_space = 0;
  --chunk_count;
  bytes_in_use -= chunk->size;
  stats.UpdateStats(chunk->size, false);

  bin_utils->InsertFreeChunkIntoBin(try_to_merge(chunk));
  Chunk* tail_chunk = prealloc_pool->top;
  if (tail_chunk != nullptr && tail_chunk == chunk) {
    high_memory_allocated_ = false;
  }
}

std::vector<std::pair<void*, size_t>> CoalescedStringentPooling::
    get_memory_info() const {
  std::vector<std::pair<void*, size_t>> regions_info;
  Chunk* chunk = prealloc_pool->start;
  regions_info.emplace_back((void*)chunk->memptr, max_pool_size - header_bytes);

  return regions_info;
}

std::pair<void*, size_t> CoalescedStringentPooling::get_tail_chunk_info()
    const {
  Chunk* tail_chunk = prealloc_pool->top;
  if (tail_chunk == nullptr) {
    return {nullptr, 0};
  }

  return {(void*)tail_chunk->memptr, tail_chunk->size};
}

std::tuple<void*, size_t, size_t> CoalescedStringentPooling::
    get_small_alloc_info() const {
  if (not small_allocs_) {
    return {nullptr, 0, 0};
  }

  return {
      small_allocs_->GetChunkPtr(),
      small_allocs_->GetkSize(),
      small_allocs_->GetkThreshold()};
}

size_t CoalescedStringentPooling::allocated_size(const void* ptr) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  if (small_allocs_ && small_allocs_->IsAllocated(ptr)) {
    return small_allocs_->Size(ptr);
  }
  auto it = chunks.find((uint64_t)ptr);
  HABANA_ASSERT(it != chunks.end());
  Chunk* chunk = it->second;
  return chunk->size;
}

void CoalescedStringentPooling::synchronize_and_free_events() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  // Synchronize on outstanding events and then free associated blocks.
  for (auto& st : hpu_events) {
    for (auto& e : st.second) {
      synEventHandle event = std::move(e.first);
      Chunk* chunk = e.second;

      auto status = synEventSynchronize(event);
      if (synStatus::synSuccess != status) {
        PT_DEVMEM_FATAL(
            Logger::formatStatusMsg(status), "Event synchronization failed");
      }

      chunk->event_count--;
      if (chunk->event_count == 0) {
        delete_chunk(chunk);
      }
      device_.get_event_handle_cache().release_handle(event);
    }
  }

  hpu_events.clear();
}

void CoalescedStringentPooling::process_events() const {
  // Process outstanding HpuEvents. Events that are completed are
  // removed from the queue, and the 'event_count' for the
  // corresponding allocation is decremented. We maintain a separate
  // list of events per stream to avoid head-of-line delays if one
  // or more streams has long-running operations.

  // Iterate over different streams.
  for (auto it = hpu_events.begin(); it != hpu_events.end();) {
    // Iterate over this stream's (event, block) pairs.
    while (!it->second.empty()) {
      auto& e = it->second.front();
      synEventHandle event = e.first;
      Chunk* chunk = e.second;

      auto status = synEventQuery(event);
      auto res = false;
      if (synStatus::synSuccess == status) {
        res = true;
      }
      if (!res) {
        e.first = event;
        break;
      } else {
        device_.get_event_handle_cache().release_handle(event);
      }

      chunk->event_count--;
      if (chunk->event_count == 0) {
        delete_chunk(chunk);
      }
      it->second.pop_front();
    }

    if (it->second.empty()) {
      it = hpu_events.erase(it);
    } else {
      it++;
    }
  }
}

void CoalescedStringentPooling::insert_events(Chunk* chunk) const {
  stream_set streams(std::move(chunk->stream_uses));
  AT_ASSERT(chunk->stream_uses.empty());
  for (auto& stream : streams) {
    // for default stream it will use compute stream
    synapse_helpers::stream& s = device_.get_stream(stream);
    synEventHandle event = device_.get_event_handle_cache().get_free_handle();
    auto status = synEventRecord(event, s);
    if (synStatus::synSuccess != status) {
      PT_DEVMEM_FATAL(
          "synStreamRecordEvent failed: ",
          status,
          " for stream::",
          s,
          " Event::",
          event);
    }

    chunk->event_count++;
    hpu_events[stream].emplace_back(event, chunk);
  }
}

bool CoalescedStringentPooling::is_stream_uses_empty(void* ptr) const {
  if ((uint64_t)ptr == 0) {
    return true;
  }
  std::unique_lock<std::mutex> lock(sp_mutex);
  if (small_allocs_ && small_allocs_->IsAllocated(ptr)) {
    return true;
  }

  auto it = chunks.find((uint64_t)ptr);
  HABANA_ASSERT(it != chunks.end());
  Chunk* chunk = it->second;
  return chunk->stream_uses.empty();
}

void CoalescedStringentPooling::record_stream(void* ptr, hpuStream_t stream)
    const {
  if ((uint64_t)ptr == 0) {
    return;
  }
  std::unique_lock<std::mutex> lock(sp_mutex);
  if (small_allocs_ && small_allocs_->IsAllocated(ptr)) {
    return;
  }

  auto it = chunks.find((uint64_t)ptr);
  HABANA_ASSERT(it != chunks.end());
  Chunk* chunk = it->second;
  HABANA_ASSERT(chunk->used);
  synapse_helpers::stream& s = device_.get_stream(stream);
  if (s == device_.get_stream(chunk->stream)) {
    // ignore uses on the allocation stream, since those don't require any
    // special synchronization
    return;
  }
  chunk->stream_uses.insert(stream);
}

CoalescedStringentPooling::SmallAllocs::SmallAllocs(
    std::unique_ptr<int8_t, std::function<void(int8_t*)>> chunk_ptr,
    size_t alignment,
    size_t bin_zero_size,
    size_t kSize)
    : chunk_ptr_(std::move(chunk_ptr)) {
  kAlignment_ = alignment;
  kSize_ = kSize;
  kThreshold_ = 2 * kAlignment_;
  HABANA_ASSERT(kSize_ % kAlignment_ == 0, "kAlignment must divide kSize");
  kUnits_ = kSize_ / kAlignment_;
  for (size_t i = kUnits_; i--;) {
    size_.push_back(0);
    map_.push_back(false);
  }
  HABANA_ASSERT(
      kThreshold_ <= bin_zero_size, "Threshold smaller then smallest bin size");
  ValidateEmpty();
}

void CoalescedStringentPooling::SmallAllocs::ValidateEmpty() const {
  const auto free_cnt = std::count(map_.cbegin(), map_.cend(), false);

  if (size_t(free_cnt) != map_.size()) {
    PT_DEVMEM_DEBUG("Some small allocations were not freed.");
  } else {
    // If empty then size_ should contain all zeros
    HABANA_ASSERT(free_cnt == std::count(size_.cbegin(), size_.cend(), 0));
  }
}

size_t CoalescedStringentPooling::SmallAllocs::UnitsOccupied() const {
  return std::count(map_.begin(), map_.end(), true);
}

void CoalescedStringentPooling::SmallAllocs::Reset() {
  if (chunk_ptr_ != nullptr) {
    ValidateEmpty();
    map_.assign(kUnits_, false);
    std::fill(size_.begin(), size_.end(), 0);
    chunk_ptr_.reset();
  }
}

CoalescedStringentPooling::SmallAllocs::~SmallAllocs() {
  Reset();
}

bool CoalescedStringentPooling::SmallAllocs::IsAllocated(
    const void* aPtr) const {
  const int8_t* const ptr = static_cast<const int8_t*>(aPtr);
  const int8_t* const chunk_ptr = chunk_ptr_.get();

  if (chunk_ptr == nullptr) {
    return false;
  }

  if (ptr < chunk_ptr) {
    return false;
  }

  if (ptr >= chunk_ptr + kSize_) {
    return false;
  }

  return map_.at(ToUnits(Offset(ptr)));
}

size_t CoalescedStringentPooling::SmallAllocs::Offset(const void* ptr) const {
  return static_cast<const int8_t*>(ptr) - chunk_ptr_.get();
}

size_t CoalescedStringentPooling::SmallAllocs::ToUnits(
    size_t offset_in_bytes) const {
  HABANA_ASSERT(offset_in_bytes % kAlignment_ == 0);
  return offset_in_bytes / kAlignment_;
}

size_t CoalescedStringentPooling::SmallAllocs::ToBytes(
    size_t offset_in_units) const {
  return offset_in_units * kAlignment_;
}

size_t CoalescedStringentPooling::SmallAllocs::Size(const void* ptr) const {
  const auto offset = ToUnits(Offset(ptr));
  HABANA_ASSERT(map_.at(offset) == true);
  return size_.at(offset);
}

void* CoalescedStringentPooling::SmallAllocs::Allocate(size_t size) {
  if (size >= kThreshold_) {
    return nullptr;
  }

  HABANA_ASSERT(chunk_ptr_.get() != nullptr);
  HABANA_ASSERT(size < kSize_);

  const auto size_in_units = ToUnits(size);
  HABANA_ASSERT(size_in_units > 0);

  const auto start =
      std::search_n(map_.begin(), map_.end(), size_in_units, false);

  if (start != map_.end()) {
    std::fill_n(start, size_in_units, true);
    const auto offset = std::distance(map_.begin(), start);

    size_[offset] = size;
    return chunk_ptr_.get() + ToBytes(offset);
  }

  return nullptr;
}

void CoalescedStringentPooling::SmallAllocs::Deallocate(const void* ptr) {
  HABANA_ASSERT(chunk_ptr_.get() != nullptr);
  const auto offset = ToUnits(Offset(ptr));

  const auto num_bytes = size_.at(offset);
  HABANA_ASSERT(num_bytes > 0);

  const auto size_in_units = ToUnits(num_bytes);
  HABANA_ASSERT(size_in_units > 0);

  auto start = map_.begin();
  std::advance(start, offset);

  {
    auto end = start;
    std::advance(end, size_in_units);
    HABANA_ASSERT(std::find(start, end, false) == end)
  }

  std::fill_n(start, size_in_units, false);

  size_[offset] = 0;
}
void* CoalescedStringentPooling::SmallAllocs::GetChunkPtr() {
  return chunk_ptr_.get();
}

std::vector<std::pair<uint64_t, uint64_t>> CoalescedStringentPooling::
    get_occupied_chunk_map() const {
  std::vector<std::pair<uint64_t, uint64_t>> occupied_chunk_map;
  std::map<uint64_t, Chunk*> chunks_ordered;
  for (auto& m : chunks) {
    chunks_ordered.insert(m);
  }
  for (auto& m : chunks_ordered) {
    auto chunk = m.second;
    if (chunk->size == 0)
      continue;
    if (chunk->used) {
      occupied_chunk_map.emplace_back(
          std::make_pair(chunk->memptr, chunk->size));
    }
  }

  return occupied_chunk_map;
}

void CoalescedStringentPooling::get_memory_mask(
    std::vector<uint64_t>& mmask) const {
  std::map<uint64_t, Chunk*> chunks_ordered;
  std::lock_guard<std::mutex> lock(sp_mutex);
  for (auto& m : chunks)
    chunks_ordered.insert(m);

  for (auto& m : chunks_ordered) {
    auto chunk = m.second;
    mmask.push_back(chunk->used ? 1 : 0);
    mmask.push_back(chunk->size);
  }
}

void CoalescedStringentPooling::get_stats(MemoryStats* mem_stats) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);

  // Fragmentation info
  bool log_fragmentation_info =
      GET_ENV_FLAG_NEW(PT_HPU_POOL_LOG_FRAGMENTATION_INFO);

  if (log_fragmentation_info) {
    const std::string occupancy_mask = "[A";
    const std::string free_mask = "[F";
    int max_chunk_per_line = 15;
    std::stringstream pool_status;
    pool_status.str("");
    pool_status.clear();
    std::map<uint64_t, Chunk*> chunks_ordered;
    for (auto& m : chunks) {
      chunks_ordered.insert(m);
    }
    int occupied_chunks = 0;
    int total_chunks = 0;
    int total_extra_spaced_chunks = 0;
    int free_chunks = 0;
    uint64_t occupied_size = header_bytes;
    uint64_t total_size = header_bytes;
    uint64_t total_exta_size = 0;
    uint64_t free_chunks_size = 0;
    uint64_t cntgs_free_chunks_size = 0;
    uint64_t available_chunks_size = 0;
    uint64_t max_cntgs_free_chunks_size = 0;
    uint64_t min_chunk_size = 0;
    uint64_t max_chunk_size = 0;
    for (auto& m : chunks_ordered) {
      auto chunk = m.second;
      if (chunk->size == 0)
        continue;
      total_chunks++;
      total_size += chunk->size;
      if (min_chunk_size == 0)
        min_chunk_size = chunk->size;
      min_chunk_size = std::min(min_chunk_size, chunk->size);
      max_chunk_size = std::max(max_chunk_size, chunk->size);
      if (chunk->extra_space && chunk->used) {
        total_extra_spaced_chunks++;
        total_exta_size += chunk->extra_space;
      }
      if (!chunk->used) {
        pool_status << free_mask;
        pool_status << "-" << chunk->size;
        pool_status << "]";
        free_chunks++;
        free_chunks_size += chunk->size;
        available_chunks_size += chunk->size;
        cntgs_free_chunks_size = getContigousChunkSize(chunk);
        if (max_cntgs_free_chunks_size < cntgs_free_chunks_size) {
          max_cntgs_free_chunks_size = cntgs_free_chunks_size;
        }
        cntgs_free_chunks_size = 0;
      } else {
        occupied_chunks++;
        occupied_size += chunk->size;
        pool_status << occupancy_mask;
        pool_status << "-" << chunk->size;
        pool_status << "]";
      }
      --max_chunk_per_line;
      if (max_chunk_per_line <= 0) {
        max_chunk_per_line = 15;
        pool_status << "\n";
      }
    }
    stats.fragmentation_percent = 100 *
        (1 - ((double)max_cntgs_free_chunks_size / available_chunks_size));
    stats.total_chunks = total_chunks;
    stats.total_size = total_size;
    stats.occupied_chunks = occupied_chunks;
    stats.occupied_size = occupied_size;
    stats.free_chunks = free_chunks;
    stats.free_chunks_size = free_chunks_size;
    stats.max_cntgs_free_chunks_size = max_cntgs_free_chunks_size;
    stats.total_extra_spaced_chunks = total_extra_spaced_chunks;
    stats.total_extra_size = total_exta_size;
    stats.min_chunk_size = min_chunk_size;
    stats.max_chunk_size = max_chunk_size;

    stats.fragmentation_mask = pool_status.str();
    pool_status.str("");
    pool_status.clear();
  }

  *mem_stats = stats;
}

void CoalescedStringentPooling::clear_stats() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  stats.num_allocs = 0;
  stats.num_frees = 0;
  stats.peak_bytes_in_use = stats.bytes_in_use;
  stats.largest_alloc_size = 0;
  stats.fragmentation_percent = 0;
  stats.fragmentation_mask = "";
}

void CoalescedStringentPooling::reset_peak_mem_stats() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  stats.peak_bytes_in_use = 0;
}

} // namespace pool_allocator
} // namespace synapse_helpers
