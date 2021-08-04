/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <synapse_api.h>

#include <habana_helpers/logging.h>
#include "CoalescedStringentPoolAllocator.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"
#include "synapse_helpers/devmem_logger.h"

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
  PT_SYNHELPER_DEBUG(
      "POOL:: InsertFreeChunkIntoBin - memptr = ",
      c->memptr,
      ", bin_index = ",
      c->bin_index);
  if (!c->used && (c->bin_index == kInvalidBinNum)) {
    uint64_t bin_index = BinIndexForSize(c->size);
    Bin* new_bin = BinFromIndex(bin_index);
    c->bin_index = bin_index;
    new_bin->free_chunks.insert(c);
  } else {
    PT_SYNHELPER_DEBUG(
        "incorrect chunk in use", c->used, "with bin index", c->bin_index);
  }
}

void BinUtils::RemoveFreeChunkFromBin(Chunk* c) const {
  PT_SYNHELPER_DEBUG(
      "POOL:: RemoveFreeChunkFromBin - memptr = ",
      c->memptr,
      ", bin_index = ",
      c->bin_index);
  if (!c->used && (c->bin_index != kInvalidBinNum)) {
    int count = BinFromIndex(c->bin_index)->free_chunks.erase(c);
    if (count < 0) {
      PT_SYNHELPER_DEBUG("could not find chunk in bin");
    } else {
      c->bin_index = kInvalidBinNum;
    }
  } else {
    PT_SYNHELPER_DEBUG(
        "incorrect chunk in use", c->used, "with bin index", c->bin_index);
  }
}

void BinUtils::RemoveFreeChunkIterFromBin(
    Bin::FreeChunkSet* free_chunks,
    const Bin::FreeChunkSet::iterator& citer) const {
  Chunk* c = *citer;
  PT_SYNHELPER_DEBUG(
      "POOL:: RemoveFreeChunkIterFromBin - memptr = ",
      c->memptr,
      ", bin_index = ",
      c->bin_index);
  HABANA_ASSERT(!c->used && (c->bin_index != kInvalidBinNum));
  free_chunks->erase(citer);
  c->bin_index = kInvalidBinNum;
}

CoalescedStringentPooling::CoalescedStringentPooling(
    uint64_t max_count,
    bool enable_merge) {
  pool_id = 0;
  chunk_count = 0;
  allocted_chunk_size = 0;
  bytes_in_use = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = DEFAULT_POOL_SIZE;
  prealloc_pool = nullptr;
  bin_utils = new BinUtils();
  small_allocs_ = nullptr;
  max_merge_count = max_count;
  enable_lfu_merging = enable_merge;
}

CoalescedStringentPooling::~CoalescedStringentPooling() {
  if (small_allocs_) {
    small_allocs_->Reset();
    small_allocs_ = nullptr;
  }
}

bool CoalescedStringentPooling::pool_create(synDeviceId deviceID, uint64_t size)
    const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  synStatus status{synStatus::synSuccess};
  size = block_align(size);
  pool_id = deviceID;
  uint64_t free_mem, total_mem;
  status = synDeviceGetMemoryInfo(deviceID, &free_mem, &total_mem);
  if (synStatus::synSuccess != status) {
    PT_SYNHELPER_DEBUG(
        "POOL:: Cannot obtain device memory info. Status: ", status);
  }

  // try to take max free memory when not set by user
  if ((size > free_mem) || (size == DEFAULT_POOL_SIZE)) {
    // setting the pool size to 99% of available memory in case of failure
    size = 0.99 * free_mem;
    PT_SYNHELPER_DEBUG(
        "POOL:: use 99% of freepool size, free mem :: ",
        free_mem,
        " size used for pool :: ",
        size);
  }
  max_pool_size = size;

  auto p = new simple_coalesced_pool_t();
  if (!p) {
    PT_SYNHELPER_DEBUG("POOL:: Cannot obtain pool memory");
    return false;
  }

  status = synDeviceMalloc(pool_id, size, 0, 0, &p->basememptr);
  if (synStatus::synSuccess != status) {
    delete (p);
    PT_SYNHELPER_FATAL(
        "POOL:: Cannot obtain device memory size. Status: ", status);
    return false;
  }

  // 0x80 bytes left for future use - header maintence in device memory instead
  // of host
  p->memptr = p->basememptr + 0x80;
  p->next = p->memptr;
  p->end = p->basememptr + size;
  p->start = nullptr;
  p->top = p->start;
  PT_SYNHELPER_DEBUG("POOL:: static coalesced stringent pool created");
  print_device_memory_stats(pool_id);
  prealloc_pool = p;

  // CoalescedStringentPooling::print_pool_stats();
  PT_SYNHELPER_DEBUG(
      "POOL:: Pool Created :: base host :: ",
      p,
      " base ptr :: ",
      p->basememptr,
      " memptr :: ",
      p->memptr,
      " prealloc_pool :: ",
      prealloc_pool->memptr,
      " end :: ",
      p->end,
      " next :: ",
      p->next);

  log_DRAM_start(p->memptr);
  log_DRAM_size(max_pool_size);
  // We create bins to fit all possible ranges that cover the
  // max_pool_size starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  for (uint64_t b = 0; b < kNumBins; b++) {
    size_t bin_size = bin_utils->BinNumToSize(b);
    PT_SYNHELPER_DEBUG("Creating bin of max chunk size ", bin_size);
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
  chunk->size = max_pool_size - 0x80;
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
  stats.bytes_in_use += 0x80;
  const auto chunk_ptr = static_cast<int8_t*>(alloc_chunk(SmallAllocs::kSize));
  const auto free_chunk = [this](int8_t* ptr) { delete_chunk(ptr); };
  small_allocs_ = absl::make_unique<SmallAllocs>(
      std::unique_ptr<int8_t, std::function<void(int8_t*)>>(
          chunk_ptr, free_chunk));
  stats.num_allocs = 0;
  return true;
}

void CoalescedStringentPooling::pool_destroy() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  simple_coalesced_pool_t* s_pool = prealloc_pool;

  CoalescedStringentPooling::print_pool_stats();

  if ((s_pool) && (chunk_count != 0)) {
    PT_SYNHELPER_DEBUG("POOL:: warning -- active chunks !!");
    PT_SYNHELPER_DEBUG("POOL:: total active chunks :: ", chunk_count);
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
          // TORCH_HABANA_CHECK(status, "synDeviceFree failed");
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
    delete bin_utils;
    bin_utils = nullptr;
    PT_SYNHELPER_DEBUG("POOL:: static coalesced pool destroyed");
  }
}

bool CoalescedStringentPooling::is_mem_threshold_hit() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  if (bytes_in_use > (max_pool_size * 0.8))
    return true;
  return false;
}

// Returns a pointer to an underlying allocated chunk of size 'num_bytes'.
void* CoalescedStringentPooling::FindChunkPtr(
    uint64_t bin_index,
    size_t num_bytes) const {
  // First identify the first bin that could satisfy num_bytes.
  for (; bin_index < kNumBins; bin_index++) {
    // Start searching from the first bin for the smallest chunk that fits
    // num_bytes.
    Bin* b = bin_utils->BinFromIndex(bin_index);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      Chunk* chunk = *citer;
      HABANA_ASSERT(!chunk->used);
      if (chunk->size >= num_bytes) {
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
             (num_bytes < DEFRAGMENT_TH(chunk->size)))) {
          try_splitting_chunks(chunk, num_bytes);
        }
        bin_utils->InsertFreeChunkIntoBin(chunk);

        PT_SYNHELPER_DEBUG("Returning: ", chunk->memptr);

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

  for (auto& m : chunks) {
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
        PT_SYNHELPER_DEBUG(
            "POOL:: can be merged :: chunk :: ",
            chunk->memptr,
            " with prev :: ",
            chunk->prev->memptr);
      }
      if (chunk->next && !chunk->next->used && chunk->next->size) {
        PT_SYNHELPER_DEBUG(
            "POOL:: can be merged :: chunk :: ",
            chunk->memptr,
            " with next :: ",
            chunk->next->memptr);
      }
    }
  }
  PT_SYNHELPER_DEBUG("POOL:: total_chunks in the pool :: ", total_chunks);
  PT_SYNHELPER_DEBUG("POOL:: total_size in the pool :: ", total_size);
  PT_SYNHELPER_DEBUG("POOL:: occupied_chunks in the pool :: ", occupied_chunks);
  PT_SYNHELPER_DEBUG(
      "POOL:: occupied_chunks size in the pool :: ", occupied_size);
  PT_SYNHELPER_DEBUG("POOL:: free chunks in the pool :: ", free_chunks);
  PT_SYNHELPER_DEBUG(
      "POOL:: free chunks size in the pool :: ", free_chunks_size);
  PT_SYNHELPER_DEBUG(
      "POOL:: max contigous chunks size in the pool :: ",
      max_cntgs_free_chunks_size);
  PT_SYNHELPER_DEBUG(
      "POOL:: total_extra_spaced_chunks in the pool  :: ",
      total_extra_spaced_chunks);
  PT_SYNHELPER_DEBUG(
      "POOL:: total_extra_size in the pool chunks :: ", total_exta_size);
  PT_SYNHELPER_DEBUG("POOL::{}", pool_status.str());
  PT_SYNHELPER_DEBUG(
      "POOL::Fragmentation = ",
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

Chunk* CoalescedStringentPooling::get_any_available_free_chunk(
    uint64_t size) const {
  Bin* bin = bin_utils->BinForSize(size);
  for (auto citer = bin->free_chunks.begin(); citer != bin->free_chunks.end();
       ++citer) {
    Chunk* chunk = *citer;
    HABANA_ASSERT(!chunk->used);

    if (chunk->size >= size) {
      PT_SYNHELPER_DEBUG(
          "POOL:: Return bigger chunk :: ",
          chunk,
          " chunk size :: ",
          chunk->size,
          " requested size :: ",
          size);

      return chunk;
    }
  }
  PT_SYNHELPER_DEBUG("POOL:: no bigger chunks !!");
  return nullptr;
}

Chunk* CoalescedStringentPooling::get_free_chunk(uint64_t size) const {
  int bin_index = bin_utils->BinIndexForSize(size);
  Chunk* new_chunk = (Chunk*)FindChunkPtr(bin_index, size);
  return new_chunk;
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
  PT_SYNHELPER_DEBUG(
      "POOL:: ctgs_chunks_size available :: ",
      ctgs_chunks_size,
      " ctgs_chunks :: ",
      ctgs_chunks,
      " chunk->size :: ",
      chunk->size);

  return ctgs_chunks_size;
}

Chunk* CoalescedStringentPooling::try_defragmenting(uint64_t size) const {
  if (!chunks_to_merge.empty()) {
    if (!defragment_chunks(size)) {
      PT_SYNHELPER_DEBUG("no chunks found for requested size after merge");
      return nullptr;
    }
  }
  auto free_chunk = get_free_chunk(size);
  if (free_chunk == nullptr) {
    PT_SYNHELPER_DEBUG(
        "POOL:: no more reusable chunk after defragment: extend pool !!");
    return nullptr;
  }
  PT_SYNHELPER_DEBUG(
      "POOL:: reusing chunk after defragment:: ",
      free_chunk->memptr,
      " req size :: ",
      size,
      " chunk size :: ",
      free_chunk->size);
  return free_chunk;
}

Chunk* CoalescedStringentPooling::reuse_chunks(uint64_t size) const {
  auto free_chunk = get_free_chunk(size);
  if (free_chunk == nullptr) {
    PT_SYNHELPER_DEBUG(
        "POOL:: no more reusable chunk: defragment or extend !!");
    return nullptr;
  }
  PT_SYNHELPER_DEBUG(
      "POOL:: reusing chunk :: ",
      free_chunk->memptr,
      " req size :: ",
      size,
      " chunk size :: ",
      free_chunk->size);
  bin_utils->RemoveFreeChunkFromBin(free_chunk);
  free_chunk->used = true;
  return free_chunk;
}

Chunk* CoalescedStringentPooling::try_block_splitting(uint64_t size) const {
  Chunk* chunk = nullptr;
  chunk = get_any_available_free_chunk(size);
  if (chunk) {
    PT_SYNHELPER_DEBUG(
        "Get any available free chunk:: ", chunk, " of Size:: ", chunk->size);
    bin_utils->RemoveFreeChunkFromBin(chunk);
    try_splitting_chunks(chunk, size);
    bin_utils->InsertFreeChunkIntoBin(chunk);
    PT_SYNHELPER_DEBUG(
        "After split chunk:: ", chunk, " of Size:: ", chunk->size);
    return chunk;
  }
  return nullptr;
}

void* CoalescedStringentPooling::extend_high_memory_allocation(
    uint64_t size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  size = block_align(size);
  size_t current_ws_size = 0;
  if (size > max_pool_size) {
    PT_SYNHELPER_DEBUG("POOL:: alloc size exceeds max size !!");
    return nullptr;
  }

  // get tail chunk
  if (prealloc_pool == nullptr) {
    PT_SYNHELPER_FATAL("POOL:: alloc invalid pool !!");
  }
  Chunk* tail_chunk = prealloc_pool->top;

  if (tail_chunk == nullptr) {
    PT_SYNHELPER_FATAL("POOL:: extend_high_memory_alloc tail chunk invalid!!");
  }

  // high memory is not allocated and tail chunk used
  if (!high_memory_allocated_ && tail_chunk->used) {
    PT_SYNHELPER_DEBUG(
        "POOL:: no space for high memory allocation, already allocated!!");
    return nullptr;
  }

  // if high_memory is already allocated, check the size with the requested size
  if (high_memory_allocated_ && (size <= tail_chunk->size)) {
    PT_SYNHELPER_DEBUG(
        "POOL:: no need to extend high memory allocation current size::",
        tail_chunk->size,
        " Requested Size::",
        size);
    return (void*)tail_chunk->memptr;
  }

  // merge lfu chunks if any
  if (!chunks_to_merge.empty()) {
    // Merge timestamped chunks whose counts have become safe for general use.
    defragment_chunks(0);
    tail_chunk = prealloc_pool->top;
  }

  // if high memory is already allocated, extend the remaining memory for the
  // requested size
  if (high_memory_allocated_) {
    current_ws_size = tail_chunk->size;
    tail_chunk->used = false;
    bin_utils->InsertFreeChunkIntoBin(try_to_merge(tail_chunk, false));
    tail_chunk = prealloc_pool->top;
  }

  if (tail_chunk->size < size) {
    PT_SYNHELPER_DEBUG(
        "POOL:: out of memory, when trying to extend high meory for size::",
        size);
    return nullptr;
  }

  // mark high_memory allocated to true and check if we need to split
  high_memory_allocated_ = true;
  auto size_left = tail_chunk->size - size;

  if (tail_chunk->bin_index == kInvalidBinNum) {
    PT_SYNHELPER_FATAL("POOL:: extend_high_memory_alloc tail chunk invalid!!");
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

  stats.UpdateStats((tail_chunk->size - current_ws_size), true);
  stats.scratch_mem_in_use = tail_chunk->size;
  return (void*)tail_chunk->memptr;
}

void* CoalescedStringentPooling::pool_alloc_chunk(
    uint64_t size,
    UNUSED bool is_workspace) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  return alloc_chunk(size);
}

void* CoalescedStringentPooling::alloc_chunk(uint64_t size) const {
  size = block_align(size);

  void* ptr = nullptr;
  if (small_allocs_)
    ptr = small_allocs_->Allocate(size);
  if (ptr != nullptr) {
    ++stats.num_allocs;
    return ptr;
  }

  if (size > max_pool_size) {
    PT_SYNHELPER_DEBUG("POOL:: alloc size exceeds max size !!");
    return nullptr;
  }
  simple_coalesced_pool_t* p = (simple_coalesced_pool_t*)prealloc_pool;
  if (prealloc_pool != p) {
    PT_SYNHELPER_FATAL("POOL:: alloc unknown pool !!");
  }

  PT_SYNHELPER_DEBUG(
      "POOL:: pool_alloc_chunk request in pool :: ", p, " for size :: ", size);
  auto old_chunk = reuse_chunks(size);
  if (old_chunk) {
    ++chunk_count;
    auto prevptr = old_chunk->prev ? old_chunk->prev->memptr : 0;
    auto nextptr = old_chunk->next ? old_chunk->next->memptr : 0;
    // extra space available in blocks after split/coalasce
    old_chunk->extra_space = old_chunk->size - size;
    PT_SYNHELPER_DEBUG(
        "POOL:: pool_alloc_chunk allocated reuse chunk :: base:: ",
        old_chunk,
        " chunk memptr ::",
        old_chunk->memptr,
        " prev :: ",
        prevptr,
        " next :: ",
        nextptr,
        " requested size :: ",
        size,
        " chunk size :: ",
        old_chunk->size,
        " extra space :: ",
        old_chunk->extra_space);
    bytes_in_use += old_chunk->size;
    stats.UpdateStats(old_chunk->size, true);
    return (void*)old_chunk->memptr;
  }

  if (!chunks_to_merge.empty()) {
    // Merge chunks whose counts have become safe for general use.
    defragment_chunks(0);
  }

  auto defrag_chunk = try_defragmenting(size);
  if (defrag_chunk) {
    ++chunk_count;
    /* remove from pool */
    Bin* bin = bin_utils->BinForSize(size);
    auto it = bin->free_chunks.find(defrag_chunk);
    if (it != bin->free_chunks.end()) {
      bin->free_chunks.erase(it);
    }
    defrag_chunk->used = true;
    chunks[defrag_chunk->memptr] = defrag_chunk;
    bytes_in_use += defrag_chunk->size;
    stats.UpdateStats(defrag_chunk->size, true);
    return (void*)defrag_chunk->memptr;
  }
  auto split_chunk = try_block_splitting(size);
  if (split_chunk) {
    ++chunk_count;
    /* remove from pool */
    Bin* bin = bin_utils->BinForSize(size);
    auto it = bin->free_chunks.find(split_chunk);
    if (it != bin->free_chunks.end()) {
      bin->free_chunks.erase(it);
    }
    split_chunk->used = true;
    chunks[split_chunk->memptr] = split_chunk;
    bytes_in_use += split_chunk->size;
    stats.UpdateStats(split_chunk->size, true);
    return (void*)split_chunk->memptr;
  }
  print_device_memory_stats(pool_id);
  print_pool_stats();
  PT_SYNHELPER_DEBUG("POOL:: pool exhausted !! for size :: ", size);
  return nullptr;
}

void CoalescedStringentPooling::try_splitting_chunks(
    Chunk* chunk,
    uint64_t size) const {
  PT_SYNHELPER_DEBUG(
      "split chunk::",
      chunk,
      " Prev:: ",
      (chunk->prev ? chunk->prev->memptr : 0),
      " memptr:: ",
      chunk->memptr,
      " next:: ",
      (chunk->next ? chunk->next->memptr : 0),
      " size:: ",
      chunk->size);

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
  chunks[new_chunk->memptr] = new_chunk;

  PT_SYNHELPER_DEBUG(
      "new chunk::",
      new_chunk,
      " prev:: ",
      (new_chunk->prev ? new_chunk->prev->memptr : 0),
      " mmeptr:: ",
      new_chunk->memptr,
      " next:: ",
      (new_chunk->next ? new_chunk->next->memptr : 0));
  PT_SYNHELPER_DEBUG(
      "modified chunk::",
      chunk,
      " prev:: ",
      (chunk->prev ? chunk->prev->memptr : 0),
      " memptr:: ",
      chunk->memptr,
      " next::",
      (chunk->next ? chunk->next->memptr : 0));
}

void CoalescedStringentPooling::merge(Chunk* c1, Chunk* c2) const {
  PT_SYNHELPER_DEBUG(
      "Merge C1::",
      c1,
      " prev:: ",
      (c1->prev ? c1->prev->memptr : 0),
      " memptr:: ",
      c1->memptr,
      " next:: ",
      (c1->next ? c1->next->memptr : 0));
  PT_SYNHELPER_DEBUG(
      "Merge C2::",
      c2,
      " prev:: ",
      (c2->prev ? c2->prev->memptr : 0),
      " memptr:: ",
      c2->memptr,
      " next:: ",
      (c2->next ? c2->next->memptr : 0));
  if (c1->used || c2->used) {
    PT_SYNHELPER_FATAL(" Chunk is in use, cannot merge ");
  }

  if (c2->prev != c1) {
    PT_SYNHELPER_FATAL(
        "Invalid c2 prev pointer prev->",
        c2->prev->memptr,
        " not equal to c1::",
        c1->memptr);
  }
  // check if c1 and c2 address are contigous(addtional check)
  if ((c1->memptr + c1->size) != c2->memptr) {
    PT_SYNHELPER_FATAL(
        "c1 & c2 are not contigous c1->memptr:: ",
        c1->memptr,
        " c2-?memptr:",
        c2->memptr);
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

  PT_SYNHELPER_DEBUG(
      "Merged Chunk C1::",
      c1,
      " prev:: ",
      (c1->prev ? c1->prev->memptr : 0),
      " memptr:: ",
      c1->memptr,
      " next:: ",
      (c1->next ? c1->next->memptr : 0));
}

Chunk* CoalescedStringentPooling::try_to_merge(Chunk* c, bool ignore_freed)
    const {
  if ((!ignore_freed) && c->freed_counter > 0)
    return c;
  Chunk* coalesced_chunk = c;

  // If the next chunk is free, merge it into c and delete it.
  if (c->next != nullptr && !(c->next)->used) {
    Chunk* new_chunk = c->next;
    if ((new_chunk->freed_counter == 0) || ignore_freed) {
      bin_utils->RemoveFreeChunkFromBin(c->next);
      merge(c, c->next);
    }
  }

  // If the previous chunk is free, merge c into it and delete c.
  if (c->prev != nullptr && !(c->prev)->used) {
    Chunk* new_chunk = c->prev;
    if ((new_chunk->freed_counter == 0) || ignore_freed) {
      coalesced_chunk = c->prev;
      bin_utils->RemoveFreeChunkFromBin(c->prev);
      merge(c->prev, c);
    }
  }

  return coalesced_chunk;
}

bool CoalescedStringentPooling::defragment_chunks(uint64_t size) const {
  bool isFreeBlockAvailble = (size == 0);
  std::list<uint64_t> to_merge;
  std::deque<Chunk*> new_chunks_to_merge;
  while (!chunks_to_merge.empty()) {
    Chunk* c = chunks_to_merge.front();
    chunks_to_merge.pop_front();
    // Make sure chunk has not already been merged
    if (c->used || (c->bin_index == kInvalidBinNum)) {
      continue;
    }
    if (c->freed_counter == 0) {
      to_merge.push_back(c->memptr);
      continue;
    }

    HABANA_ASSERT(c->bin_index != kInvalidBinNum);
    if (c->freed_counter < max_merge_count) {
      c->freed_counter = 0;
      to_merge.push_back(c->memptr);
    } else if (size > 0) {
      to_merge.push_back(c->memptr);
    } else {
      new_chunks_to_merge.push_back(c);
    }
  }
  HABANA_ASSERT(chunks_to_merge.empty());
  std::swap(chunks_to_merge, new_chunks_to_merge);

  // All candidate chunks have been moved from chunks_to_merge to to_merge.
  // size == 0  : standard merge, merge them all,
  // otherwise  : merge just until a Chunk of the required size is produced.
  for (auto& ptr : to_merge) {
    auto it = chunks.find(ptr);
    if (it == chunks.end())
      continue;
    Chunk* c = it->second;

    if (size == 0 || !isFreeBlockAvailble) {
      HABANA_ASSERT(c->bin_index != kInvalidBinNum);
      HABANA_ASSERT(!c->used);
      bin_utils->RemoveFreeChunkFromBin(c);
      Chunk* new_chunk = try_to_merge(c, (size > 0));
      bin_utils->InsertFreeChunkIntoBin(new_chunk);
      if (size > 0) {
        if (new_chunk->memptr != c->memptr && new_chunk->freed_counter > 0) {
          chunks_to_merge.push_back(new_chunk);
        }
        if (new_chunk->size >= size) {
          isFreeBlockAvailble = true;
        }
      }
    } else {
      // We were force merging Chunks, but managed
      // to create a satisfying Chunk so just requeue the rest.
      chunks_to_merge.push_back(c);
    }
  }
  return isFreeBlockAvailble;
}

void CoalescedStringentPooling::pool_free_chunk(void* ptr) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  delete_chunk(ptr);
}

void CoalescedStringentPooling::delete_chunk(void* ptr) const {
  if ((uint64_t)ptr == 0) {
    PT_SYNHELPER_DEBUG("POOL:: null ptr");
    return;
  }

  if (small_allocs_ && small_allocs_->IsAllocated(ptr)) {
    small_allocs_->Deallocate(ptr);
    ++stats.num_frees;
    return;
  }

  auto it = chunks.find((uint64_t)ptr);
  HABANA_ASSERT(it != chunks.end());
  Chunk* chunk = it->second;
  chunk->used = false;
  chunk->bin_index = kInvalidBinNum;
  chunk->extra_space = 0;
  --chunk_count;
  bytes_in_use -= chunk->size;
  stats.UpdateStats(chunk->size, false);

  if (enable_lfu_merging) {
    chunk->freed_counter += 1;
    bin_utils->InsertFreeChunkIntoBin(chunk);
    chunks_to_merge.push_back(chunk);
  } else {
    bin_utils->InsertFreeChunkIntoBin(try_to_merge(chunk, false));
  }
}

CoalescedStringentPooling::SmallAllocs::SmallAllocs(
    std::unique_ptr<int8_t, std::function<void(int8_t*)>> chunk_ptr)
    : chunk_ptr_(std::move(chunk_ptr)), map_(kUnits, false), size_{0} {
  static_assert(
      kThreshold <= BinUtils::BinNumToSize(0),
      "Threshold smaller then smallest bin size");
  ValidateEmpty();
}

void CoalescedStringentPooling::SmallAllocs::ValidateEmpty() const {
  const auto free_cnt = std::count(map_.cbegin(), map_.cend(), false);

  if (size_t(free_cnt) != map_.size()) {
    PT_SYNHELPER_DEBUG("Some small allocations were not freed.");
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
    map_.assign(kUnits, false);
    size_.fill(0);
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

  if (ptr >= chunk_ptr + kSize) {
    return false;
  }

  return map_.at(ToUnits(Offset(ptr)));
}

size_t CoalescedStringentPooling::SmallAllocs::Offset(const void* ptr) const {
  return static_cast<const int8_t*>(ptr) - chunk_ptr_.get();
}

size_t CoalescedStringentPooling::SmallAllocs::ToUnits(size_t offset_in_bytes) {
  HABANA_ASSERT(offset_in_bytes % kAlignment == 0);
  return offset_in_bytes / kAlignment;
}

size_t CoalescedStringentPooling::SmallAllocs::ToBytes(size_t offset_in_units) {
  return offset_in_units * kAlignment;
}

size_t CoalescedStringentPooling::SmallAllocs::Size(const void* ptr) const {
  const auto offset = ToUnits(Offset(ptr));
  HABANA_ASSERT(map_.at(offset) == true);
  return size_.at(offset);
}

void* CoalescedStringentPooling::SmallAllocs::Allocate(size_t size) {
  if (size >= kThreshold) {
    return nullptr;
  }

  HABANA_ASSERT(chunk_ptr_.get() != nullptr);
  HABANA_ASSERT(size < kSize);

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

void CoalescedStringentPooling::get_stats(MemoryStats* mem_stats) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  // Fragmentation info
  bool log_fragmentation_info =
      GET_ENV_FLAG(PT_HPU_POOL_LOG_FRAGMENTATION_INFO);

  if (log_fragmentation_info) {
    const std::string occupancy_mask = "[+++]";
    const std::string free_mask = "[00000]";
    std::stringstream pool_status;
    pool_status.str("");
    pool_status.clear();
    std::map<uint64_t, Chunk*> chunks_ordered;
    for (auto& m : chunks) {
      chunks_ordered.insert(m);
    }
    uint64_t cntgs_free_chunks_size = 0;
    uint64_t available_chunks_size = 0;
    uint64_t max_cntgs_free_chunks_size = 0;
    for (auto& m : chunks_ordered) {
      auto chunk = m.second;
      if (!chunk->used && (chunk->size != 0)) {
        pool_status << free_mask;
        available_chunks_size += chunk->size;
        cntgs_free_chunks_size = getContigousChunkSize(chunk);
        if (max_cntgs_free_chunks_size < cntgs_free_chunks_size) {
          max_cntgs_free_chunks_size = cntgs_free_chunks_size;
        }
        cntgs_free_chunks_size = 0;
      } else {
        pool_status << occupancy_mask;
      }
    }
    stats.fragmentation_percent = 100 *
        (1 - ((double)max_cntgs_free_chunks_size / available_chunks_size));
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

} // namespace pool_allocator
} // namespace synapse_helpers
