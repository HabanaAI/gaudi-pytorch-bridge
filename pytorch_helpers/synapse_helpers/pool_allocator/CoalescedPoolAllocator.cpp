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
#include "CoalescedPoolAllocator.h"
#include "synapse_helpers/devmem_logger.h"
#include "utils.h"

#define DEFRAGMENT_TH(arg) std::ceil(0.9 * (arg))
//#define DEFRAGMENT_ON_REUSE

namespace synapse_helpers {
namespace pool_allocator {

StaticCoalescedPooling::StaticCoalescedPooling() {
  pool_id = 0;
  chunk_count = 0;
  allocted_chunk_size = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = DEFAULT_POOL_SIZE;
  prealloc_pool = nullptr;
}

bool StaticCoalescedPooling::pool_create(synDeviceId deviceID, uint64_t size)
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
    // setting the pool size to 90% of available memory in case of failure
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
  PT_SYNHELPER_DEBUG("POOL:: static coalesced pool created");
  print_device_memory_stats(pool_id);
  prealloc_pool = p;

  // StaticCoalescedPooling::print_pool_stats();
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
  return true;
}

void StaticCoalescedPooling::pool_destroy() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  simple_coalesced_pool_t* s_pool = prealloc_pool;

  StaticCoalescedPooling::print_pool_stats();

  if ((s_pool) && (chunk_count != 0)) {
    PT_SYNHELPER_DEBUG("POOL:: warning -- active chunks !!");
    PT_SYNHELPER_DEBUG("POOL:: total active chunks :: ", chunk_count);
  }

  if (s_pool) {
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

    s_pool->basememptr = 0;
    for (auto& m : chunks) {
      delete (m.second);
    }
    chunks.clear();
    free_list.clear();
    delete (s_pool);
    s_pool = nullptr;
    PT_SYNHELPER_DEBUG("POOL:: static coalesced pool destroyed");
  }
}

static uint64_t pool_available(simple_coalesced_pool_t* p) {
  return p->end - p->next;
}

void StaticCoalescedPooling::print_pool_stats() const {
  static const std::string occupancy_mask = "[+++]";
  static const std::string free_mask = "[00000]";
  static std::stringstream pool_status;
  static int occupied_chunks = 0;
  static int total_chunks = 0;
  static uint64_t occupied_size = 0;
  static uint64_t total_size = 0;
  static int total_extra_spaced_chunks = 0;
  static uint64_t total_exta_size = 0;
  static uint64_t cntgs_free_chunks_size = 0;
  static uint64_t max_cntgs_free_chunks_size = 0;
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

bool StaticCoalescedPooling::skip_chunk(Chunk* chunk, uint64_t size_req) const {
  // skip the chunk if
  // 1. already in use
  // 2. chunk size cannot accomodate requested size
  // 3. requested size is not 90% of the chunk size
  // this is to avoid small sized requests consuming bigger free chunks
  if (chunk->used || (chunk->size < size_req) ||
      (size_req < DEFRAGMENT_TH(chunk->size))) {
    return true;
  }
  return false;
}

Chunk* StaticCoalescedPooling::get_any_available_free_chunk(
    uint64_t size) const {
  Chunk key = Chunk(size);
  auto it = free_list.lower_bound(&key);
  if (it != free_list.end()) {
    auto chunk = *it;
    PT_SYNHELPER_DEBUG(
        "POOL:: Return bigger chunk :: ",
        chunk,
        " chunk size :: ",
        chunk->size,
        " requested size :: ",
        size);
    return chunk;
  }
  PT_SYNHELPER_DEBUG("POOL:: no bigger chunks !!");
  return nullptr;
}

Chunk* StaticCoalescedPooling::get_free_chunk(uint64_t size) const {
  Chunk key = Chunk(size);
  auto it = free_list.lower_bound(&key);

  if (it != free_list.end()) {
    auto chunk = *it;
    if (!skip_chunk(chunk, size)) {
      chunk->used = true;
      free_list.erase(chunk);
      return chunk;
    }
  }
  return nullptr;
}

Chunk* StaticCoalescedPooling::defragment_on_reuse(void* ptr, uint64_t size)
    const {
  simple_coalesced_pool_t* p = (simple_coalesced_pool_t*)ptr;
  auto curr_size = p->end - p->next;
  // do not defragment until pool touches 75% capacity
  if (curr_size >= (0.75 * max_pool_size)) {
    PT_SYNHELPER_DEBUG(
        "POOL:: more than 75% pool available, not defragmenting, \
            current size :: ",
        curr_size,
        " pool size :: ",
        max_pool_size);
    return nullptr;
  }
  return try_defragmenting(ptr, size);
}

bool StaticCoalescedPooling::isChunkContigous(Chunk* chunk1, Chunk* chunk2)
    const {
  if ((chunk1->memptr + chunk1->size) == chunk2->memptr) {
    return true;
  }
  return false;
}

uint64_t StaticCoalescedPooling::getContigousChunkSize(Chunk* chunk) const {
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

bool StaticCoalescedPooling::isContigousBlockAvailable(uint64_t size) const {
  uint64_t ctgs_chunks_size = 0;
  for (auto& chunk : free_list) {
    if (!chunk->used && (chunk->size != 0)) {
      ctgs_chunks_size = getContigousChunkSize(chunk);
      if (ctgs_chunks_size >= size) {
        PT_SYNHELPER_DEBUG(
            "POOL:: ctgs_chunks_size available for defragmentation");
        return true;
      }
    }
  }
  return false;
}

Chunk* StaticCoalescedPooling::try_defragmenting(void* ptr, uint64_t size)
    const {
  simple_coalesced_pool_t* p = (simple_coalesced_pool_t*)ptr;

  // try defragmenting the pool
  // we defragment only for the requested size as there is no
  // pre-allocated bin for different block sizes
  bool isFreeBlockAvailble = false;
  uint16_t counter = 0;
  if (free_list.empty()) {
    PT_SYNHELPER_DEBUG("POOL:: no free blocks availabe: ");
    return nullptr;
  }
  do {
    isFreeBlockAvailble = pool_defragment(size);
    counter++;
    if (isFreeBlockAvailble) {
      PT_SYNHELPER_DEBUG(
          "POOL:: free block available for use for size : ", size);
      break;
    }
    // worse case :parse the entire list once for every chunk to find free
    // blocks
    if (counter > free_list.size()) {
      PT_SYNHELPER_DEBUG(
          "POOL:: no more contigous chunks to accomodate request in the pool !");
      break;
    }

  } while ((!isFreeBlockAvailble) && (isContigousBlockAvailable(size)));

  auto free_chunk = get_free_chunk(size);
  if (free_chunk == nullptr) {
    if (isFreeBlockAvailble) {
      PT_SYNHELPER_DEBUG(
          "POOL:: Free blocks available -- block split, p->end :: ",
          p->end,
          " p->next :: ",
          p->next,
          " max_pool_size :: ",
          max_pool_size);
      isFreeBlockAvailble = false;
    }
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

Chunk* StaticCoalescedPooling::reuse_chunks(uint64_t size) const {
  auto free_chunk = get_free_chunk(size);
  if (free_chunk == nullptr) {
#ifdef DEFRAGMENT_ON_REUSE
    simple_coalesced_pool_t* p = (simple_coalesced_pool_t*)prealloc_pool;
    return defragment_on_reuse(p, size);
#else
    PT_SYNHELPER_DEBUG(
        "POOL:: no more reusable chunk: defragment or extend !!");
    return nullptr;
#endif
  }
  PT_SYNHELPER_DEBUG(
      "POOL:: reusing chunk :: ",
      free_chunk->memptr,
      " req size :: ",
      size,
      " chunk size :: ",
      free_chunk->size);
  free_chunk->used = true;
  return free_chunk;
}

Chunk* StaticCoalescedPooling::try_block_splitting(uint64_t size) const {
  Chunk* big_chunk = nullptr;
  big_chunk = get_any_available_free_chunk(size);
  if (big_chunk) {
    auto split_chunk = try_splitting_chunks(big_chunk, size);
    if (split_chunk) {
      PT_SYNHELPER_DEBUG(
          "POOL:: bigger chunk split :: big_chunk :: ",
          big_chunk,
          " big_chunk size :: ",
          big_chunk->size,
          " split chunk :: ",
          split_chunk,
          " split chunk size :: ",
          split_chunk->size);
      return split_chunk;
    }
  }
  return nullptr;
}

void* StaticCoalescedPooling::pool_alloc_chunk(uint64_t size) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  size = block_align(size);

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
    return (void*)old_chunk->memptr;
  }

  if (pool_available(p) < size) {
    // TBD: implement better algorithms
    auto defrag_chunk = try_defragmenting(p, size);
    if (defrag_chunk) {
      ++chunk_count;
      defrag_chunk->used = true;
      /* remove from pool */
      auto it = free_list.find(defrag_chunk);
      if (it != free_list.end()) {
        free_list.erase(it);
      }
      chunks[defrag_chunk->memptr] = defrag_chunk;
      return (void*)defrag_chunk->memptr;
    }
    auto split_chunk = try_block_splitting(size);
    if (split_chunk) {
      ++chunk_count;
      split_chunk->used = true;
      /* remove from pool */
      auto it = free_list.find(split_chunk);
      if (it != free_list.end()) {
        free_list.erase(it);
      }
      chunks[split_chunk->memptr] = split_chunk;
      return (void*)split_chunk->memptr;
    }
    print_device_memory_stats(pool_id);
    print_pool_stats();
    PT_SYNHELPER_DEBUG("POOL:: pool exhausted !! for size :: ", size);
    return nullptr;
  }

  // create a chunk
  Chunk* chunk = new Chunk();
  if (!chunk) {
    PT_SYNHELPER_DEBUG("POOL:: Cannot create a chunk");
    return nullptr;
  }
  chunk->memptr = (uint64_t)p->next;
  chunk->extra_space = 0;
  chunk->size = size;
  chunk->used = true;
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
  allocted_chunk_size += size;
  ++chunk_count;

  auto prevptr = chunk->prev ? chunk->prev->memptr : 0;
  auto nextptr = chunk->next ? chunk->next->memptr : 0;
  PT_SYNHELPER_DEBUG(
      "POOL:: pool_alloc_chunk allocated :: base:: ",
      chunk,
      " chunk memptr ::",
      chunk->memptr,
      " prev :: ",
      prevptr,
      " next :: ",
      nextptr,
      " requested size :: ",
      size,
      " chunk size :: ",
      chunk->size);
  PT_SYNHELPER_DEBUG("POOL:: Allocated chunk_count :: ", chunk_count);

  chunks[chunk->memptr] = chunk;
  return (void*)chunk->memptr;
}

bool StaticCoalescedPooling::canMergeNextChunk(Chunk* chunk, uint64_t size)
    const {
  return (
      !chunk->used && chunk->next && !chunk->next->used && chunk->next->size &&
      ((chunk->size + chunk->next->size) >= size));
}

Chunk* StaticCoalescedPooling::mergeNextChunk(Chunk* chunk) const {
  auto adj_chunk = chunk->next;
  // check if current chunk and next chunk addresses are contigous
  if (((chunk->memptr + chunk->size) != adj_chunk->memptr) ||
      (chunk->next->used)) {
    PT_SYNHELPER_DEBUG("POOL:: Next chunk is not contigous or free ");
    return nullptr;
  }

  /* remove from pool*/
  auto it = free_list.find(chunk);
  if (it != free_list.end()) {
    free_list.erase(it);
  }
  chunk->next = adj_chunk->next;

  auto prevptr = chunk->prev ? chunk->prev->memptr : 0;
  auto nextptr = chunk->next ? chunk->next->memptr : 0;
  PT_SYNHELPER_DEBUG(
      "POOL:: ***Merged - next** chunk :: ",
      chunk->memptr,
      " prev :: ",
      prevptr,
      " next :: ",
      nextptr,
      " merged with :: ",
      adj_chunk->memptr);
  PT_SYNHELPER_DEBUG(
      "POOL:: ***Merged - next** chunk size :: ",
      chunk->size + adj_chunk->size,
      " old size :: ",
      chunk->size);

  // assuming device memory is contingous
  chunk->size = chunk->size + adj_chunk->size;

  /* remove old chuk from pool */
  it = free_list.find(adj_chunk);
  if (it != free_list.end()) {
    free_list.erase(it);
  }

  /* insert merged chunk to the pool */
  free_list.insert(chunk);
  if (prealloc_pool->top == adj_chunk) {
    // update top
    prealloc_pool->top = chunk;
  }

  /* remove old chunk from chunks map*/
  auto it1 = chunks.find(adj_chunk->memptr);
  if (it1 != chunks.end()) {
    chunks.erase(it1);
  }
  adj_chunk->used = false;
  adj_chunk->extra_space = 0;
  adj_chunk->size = 0;
  adj_chunk->memptr = 0;
  adj_chunk->next = nullptr;
  adj_chunk->prev = nullptr;

  delete adj_chunk;
  return chunk;
}

bool StaticCoalescedPooling::canMergePreviousChunk(Chunk* chunk, uint64_t size)
    const {
  return (
      !chunk->used && chunk->prev && !chunk->prev->used && chunk->prev->size &&
      ((chunk->size + chunk->prev->size) >= size));
}

Chunk* StaticCoalescedPooling::mergePreviousChunk(Chunk* chunk) const {
  auto adj_chunk = chunk->prev;

  // check if current chunk and previous chunk addresses are contigous
  if (((adj_chunk->memptr + adj_chunk->size) != chunk->memptr) ||
      (chunk->prev->used)) {
    PT_SYNHELPER_DEBUG("POOL:: Previous chunk is not contigous ");
    return nullptr;
  }

  /* remove from pool*/
  auto it = free_list.find(adj_chunk);
  if (it != free_list.end()) {
    free_list.erase(it);
  }

  if (prealloc_pool->top == chunk) {
    // update top
    prealloc_pool->top = adj_chunk;
  }

  // assuming device memory is contingous
  adj_chunk->next = chunk->next;

  auto prevptr = chunk->prev ? chunk->prev->memptr : 0;
  auto nextptr = chunk->next ? chunk->next->memptr : 0;
  PT_SYNHELPER_DEBUG(
      "POOL:: ***Merged - previous** chunk :: ",
      chunk->memptr,
      " prev :: ",
      prevptr,
      " next :: ",
      nextptr,
      " merged with :: ",
      adj_chunk->memptr);
  PT_SYNHELPER_DEBUG(
      "POOL:: ***Merged - previous** chunk size :: ",
      chunk->size + adj_chunk->size,
      " old size :: ",
      chunk->size);

  adj_chunk->size = chunk->size + adj_chunk->size;

  /* insert merged chunk to the pool */
  free_list.insert(adj_chunk);

  /* remove old chuk from pool */
  it = free_list.find(chunk);
  if (it != free_list.end()) {
    free_list.erase(it);
  }
  /* remove old chunk from chunks map*/
  auto it1 = chunks.find(chunk->memptr);
  if (it1 != chunks.end()) {
    chunks.erase(it1);
  }
  chunk->used = false;
  chunk->extra_space = 0;
  chunk->size = 0;
  chunk->memptr = 0;
  chunk->next = nullptr;
  chunk->prev = nullptr;

  delete chunk;
  return adj_chunk;
}

Chunk* StaticCoalescedPooling::try_coalescing_chunks(
    Chunk* chunk,
    uint64_t size) const {
  if (canMergePreviousChunk(chunk, size)) {
    // coalesce adjacent free chunks
    chunk = mergePreviousChunk(chunk);
    return chunk;
  }
  if (canMergeNextChunk(chunk, size)) {
    // coalesce adjacent free chunks
    chunk = mergeNextChunk(chunk);
    return chunk;
  }
  if (chunk->prev && chunk->next) {
    if (!chunk->prev->used && !chunk->next->used) {
      // try merging the missed out chunks
      // chunk = mergePreviousChunk(chunk);
      chunk = mergeNextChunk(chunk);
      return chunk;
    }
  }
  return nullptr;
}

Chunk* StaticCoalescedPooling::create_chunk() const {
  // create a chunk
  Chunk* chunk = new Chunk();
  if (!chunk) {
    PT_SYNHELPER_DEBUG("POOL:: Cannot create a chunk");
    return nullptr;
  }
  chunk->memptr = 0;
  chunk->extra_space = 0;
  chunk->size = 0;
  chunk->used = false;
  chunk->next = nullptr;
  chunk->prev = nullptr;

  return chunk;
}

Chunk* StaticCoalescedPooling::try_splitting_chunks(Chunk* chunk, uint64_t size)
    const {
  auto new_size = chunk->size - size;
  auto new_memptr = chunk->memptr + new_size;
  // TBD: ensure memory is contigous
  if ((chunk->memptr + new_size) != new_memptr) {
    PT_SYNHELPER_DEBUG("POOL:: split blocks not contigous");
    return nullptr;
  }
  /* remove from pool list and insert it back as size changes */
  auto it = free_list.find(chunk);
  if (it != free_list.end()) {
    free_list.erase(it);
  }
  /* create a new chunk */
  Chunk* new_chunk = new Chunk(size, 0, false, chunk, chunk->next, new_memptr);

  if (prealloc_pool->top == chunk) {
    // update top
    prealloc_pool->top = new_chunk;
  }
  chunk->next = new_chunk;
  chunk->size = new_size;

  free_list.insert(chunk);
  free_list.insert(new_chunk);
  // insert new chunk to chunks
  chunks[new_chunk->memptr] = new_chunk;
  return new_chunk;
}

Chunk* StaticCoalescedPooling::get_nearest_chunk(uint64_t size) const {
  auto chunk_start = free_list.begin();
  auto chunk_end = free_list.end();
  Chunk key = Chunk(size);
  auto it = free_list.lower_bound(&key);

  if (it == chunk_end) {
    if (it != chunk_start)
      --it;
    auto chunk = *it;
    return chunk;
  }

  auto nt = std::next(it);

  if (nt == chunk_end)
    return *it;
  auto chunk = *it;
  auto nxt_chunk = *nt;
  return chunk->size - size < nxt_chunk->size - size ? nxt_chunk : chunk;
}

bool StaticCoalescedPooling::pool_defragment(uint64_t size) const {
  bool isFreeBlockAvailble = false;
  PT_SYNHELPER_DEBUG("POOL:: Try to coalesce and split if needed");
  auto chunk = get_nearest_chunk(size);
  if (size < DEFRAGMENT_TH(chunk->size)) {
    isFreeBlockAvailble = true;
    auto split_chunk = try_splitting_chunks(chunk, size);
    if (split_chunk) {
      PT_SYNHELPER_DEBUG(
          "POOL:: chunk splitted successfully :: newchunk :: ",
          split_chunk,
          " new chunk size :: ",
          split_chunk->size,
          " old chunk :: ",
          chunk,
          " old chunk size :: ",
          chunk->size);
    }
  } else {
    auto new_chunk = try_coalescing_chunks(chunk, size);
    if (new_chunk && new_chunk->size >= size) {
      isFreeBlockAvailble = true;
      PT_SYNHELPER_DEBUG(
          "POOL:: pool defragmentation succeeded for requested size :: ",
          size,
          " chunk->size :: ",
          new_chunk->size);
      if (size < DEFRAGMENT_TH(new_chunk->size)) {
        PT_SYNHELPER_DEBUG(
            "POOL:: try block splitting for size :: ",
            size,
            " in chunk of chunk->size :: ",
            new_chunk->size);
        auto split_chunk = try_splitting_chunks(new_chunk, size);
        if (split_chunk) {
          PT_SYNHELPER_DEBUG(
              "POOL:: chunk splitted successfully :: newchunk :: ",
              split_chunk,
              " new chunk size :: ",
              split_chunk->size,
              " old chunk :: ",
              new_chunk,
              " old chunk size :: ",
              new_chunk->size);
        }
      }
    }
  }

  return isFreeBlockAvailble;
}

void StaticCoalescedPooling::pool_free_chunk(void* ptr) const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  if ((uint64_t)ptr == 0) {
    PT_SYNHELPER_DEBUG("POOL:: null ptr");
    return;
  }

  auto it = chunks.find((uint64_t)ptr);
  HABANA_ASSERT(it != chunks.end());
  Chunk* chunk = it->second;
  chunk->used = false;
  chunk->extra_space = 0;
  free_list.insert(chunk);
  --chunk_count;
  if (chunk->prev && chunk->prev->used && isChunkContigous(chunk->prev, chunk))
    mergePreviousChunk(chunk);
  if (chunk->next && chunk->next->used && isChunkContigous(chunk, chunk->next))
    mergeNextChunk(chunk);
}

} // namespace pool_allocator
} // namespace synapse_helpers
