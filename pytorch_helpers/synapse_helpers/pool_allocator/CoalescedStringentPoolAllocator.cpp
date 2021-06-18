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
#include "CoalescedStringentPoolAllocator.h"
#include "synapse_helpers/devmem_logger.h"
#include "utils.h"

#define DEFRAGMENT_TH(arg) std::ceil(0.9 * (arg))

namespace synapse_helpers {
namespace pool_allocator {

CoalescedStringentPooling::CoalescedStringentPooling() {
  pool_id = 0;
  chunk_count = 0;
  allocted_chunk_size = 0;
  bytes_in_use = 0;
  free_chunks = 0;
  free_chunks_size = 0;
  max_pool_size = DEFAULT_POOL_SIZE;
  prealloc_pool = nullptr;
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
  PT_SYNHELPER_DEBUG("POOL:: static coalesced pool created");
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

bool CoalescedStringentPooling::is_mem_threshold_hit() const {
  const std::lock_guard<std::mutex> lock(sp_mutex);
  if (bytes_in_use > (max_pool_size * 0.8))
    return true;
  return false;
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

Chunk* CoalescedStringentPooling::get_free_chunk(uint64_t size) const {
  Chunk* new_chunk = nullptr;
  for (auto& chunk : free_list) {
    if ((chunk->size == size) ||
        (chunk->size > size && (size > DEFRAGMENT_TH(chunk->size)))) {
      new_chunk = chunk;
      break;
    }
  }
  if (new_chunk) {
    free_list.erase(new_chunk);
    new_chunk->used = true;
    return new_chunk;
  }
  return nullptr;
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

bool CoalescedStringentPooling::isContigousBlockAvailable(uint64_t size) const {
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

Chunk* CoalescedStringentPooling::try_defragmenting(void* ptr, uint64_t size)
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
  free_chunk->used = true;
  return free_chunk;
}

Chunk* CoalescedStringentPooling::try_block_splitting(uint64_t size) const {
  Chunk* big_chunk = nullptr;
  big_chunk = get_any_available_free_chunk(size);
  if (big_chunk) {
    PT_SYNHELPER_DEBUG(
        "Get any available free chunk:: ",
        big_chunk,
        " of Size:: ",
        big_chunk->size);
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

void* CoalescedStringentPooling::pool_alloc_chunk(uint64_t size) const {
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
    bytes_in_use += old_chunk->size;
    return (void*)old_chunk->memptr;
  }

  if (pool_available(p) < size) {
    // TBD: implement better algorithms
    auto defrag_chunk = try_defragmenting(p, size);
    if (defrag_chunk) {
      ++chunk_count;
      /* remove from pool */
      auto it = free_list.find(defrag_chunk);
      if (it != free_list.end()) {
        free_list.erase(it);
      }
      defrag_chunk->used = true;
      chunks[defrag_chunk->memptr] = defrag_chunk;
      bytes_in_use += defrag_chunk->size;
      return (void*)defrag_chunk->memptr;
    }
    auto split_chunk = try_block_splitting(size);
    if (split_chunk) {
      ++chunk_count;
      /* remove from pool */
      auto it = free_list.find(split_chunk);
      if (it != free_list.end()) {
        free_list.erase(it);
      }
      split_chunk->used = true;
      chunks[split_chunk->memptr] = split_chunk;
      bytes_in_use += split_chunk->size;
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
  bytes_in_use += chunk->size;
  return (void*)chunk->memptr;
}

bool CoalescedStringentPooling::canMergeNextChunk(Chunk* chunk, uint64_t size)
    const {
  return (
      !chunk->used && chunk->next && !chunk->next->used && chunk->next->size &&
      ((chunk->size + chunk->next->size) >= size));
}

bool CoalescedStringentPooling::canMergePreviousChunk(
    Chunk* chunk,
    uint64_t size) const {
  PT_SYNHELPER_DEBUG("POOL:: Next chunk is not contigous or free ", size);

  return (
      !chunk->used && chunk->prev && !chunk->prev->used && chunk->prev->size &&
      ((chunk->size + chunk->prev->size) >= size));
}

Chunk* CoalescedStringentPooling::try_splitting_chunks(
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
      (chunk->next ? chunk->next->memptr : 0));
  if (chunk->size < size) {
    PT_SYNHELPER_DEBUG(
        "POOL:: chunk cant be split, chunk is smaller. chunk size:: ",
        chunk->size,
        " split size:: ",
        size);
    return nullptr;
  }

  // Delete the old chunk before modifying the size
  auto it = free_list.find(chunk);
  if (it != free_list.end()) {
    free_list.erase(it);
  }
  Chunk* new_chunk = new Chunk();
  new_chunk->memptr = chunk->memptr + size;
  new_chunk->size = chunk->size - size;
  chunk->size = size;

  new_chunk->used = false;

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

  free_list.insert(chunk);
  free_list.insert(new_chunk);
  // insert new chunk to chunks
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
  return chunk;
}

Chunk* CoalescedStringentPooling::merge(Chunk* c1, Chunk* c2) const {
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
    PT_SYNHELPER_DEBUG(" Chunk is in use, cannot merge ");
    return nullptr;
  }

  if (c2->prev != c1) {
    PT_SYNHELPER_FATAL(
        "Invalid c2 prev pointer prev->",
        c2->prev->memptr,
        " not equal to c1::",
        c1->memptr);
    return nullptr;
  }
  // check if c1 and c2 address are contigous(addtional check)
  if ((c1->memptr + c1->size) != c2->memptr) {
    PT_SYNHELPER_FATAL(
        "c1 & c2 are not contigous c1->memptr:: ",
        c1->memptr,
        " c2-?memptr:",
        c2->memptr);
    return nullptr;
  }

  if (prealloc_pool->top == c2) {
    // update top
    prealloc_pool->top = c1;
  }

  auto it = free_list.find(c1);
  if (it != free_list.end()) {
    free_list.erase(it);
  }
  it = free_list.find(c2);
  if (it != free_list.end()) {
    free_list.erase(it);
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

  free_list.insert(c1);

  PT_SYNHELPER_DEBUG(
      "Merged Chunk C1::",
      c1,
      " prev:: ",
      (c1->prev ? c1->prev->memptr : 0),
      " memptr:: ",
      c1->memptr,
      " next:: ",
      (c1->next ? c1->next->memptr : 0));
  return c1;
}

bool CoalescedStringentPooling::merge_chunks(
    std::list<uint64_t> ptrs,
    bool merge_nxt,
    uint64_t size) const {
  bool isFreeBlockAvailble = false;

  for (auto& ptr : ptrs) {
    auto it = chunks.find(ptr);
    if (it == chunks.end()) // in some cases chunk would have been merged, so it
                            // wont be in the map
      continue;
    Chunk* chunk = it->second;
    Chunk* new_chunk = nullptr;
    if (merge_nxt) {
      if (chunk->next && !chunk->next->used)
        new_chunk = merge(chunk, chunk->next);
    } else {
      if (chunk->prev && !chunk->prev->used)
        new_chunk = merge(chunk->prev, chunk);
    }

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
      break;
    }
  }
  return isFreeBlockAvailble;
}

bool CoalescedStringentPooling::pool_defragment(uint64_t size) const {
  bool isFreeBlockAvailble = false;
  PT_SYNHELPER_DEBUG("POOL:: Try to coalesce and split if needed");
  std::list<uint64_t> to_merge;
  // check previous chunks
  for (auto& chunk : free_list) {
    if (canMergeNextChunk(chunk, size)) {
      // coalesce adjacent free chunks
      to_merge.push_front(chunk->memptr);
    }
  }
  if (!to_merge.empty())
    isFreeBlockAvailble = merge_chunks(to_merge, true, size);

  // check next chunks
  if (!isFreeBlockAvailble) {
    to_merge.clear();
    for (auto& chunk : free_list) {
      if (canMergePreviousChunk(chunk, size)) {
        // coalesce adjacent free chunks
        to_merge.push_front(chunk->memptr);
      }
    }
    if (!to_merge.empty())
      isFreeBlockAvailble = merge_chunks(to_merge, false, size);
  }

  // try merge the left out chunks irrespective of size
  if (!isFreeBlockAvailble) {
    to_merge.clear();
    for (auto& chunk : free_list) {
      if (chunk->next && !chunk->next->used) {
        to_merge.push_front(chunk->memptr);
      }
    }
    if (!to_merge.empty())
      isFreeBlockAvailble = merge_chunks(to_merge, true, size);
  }

  if (!isFreeBlockAvailble) {
    to_merge.clear();
    for (auto& chunk : free_list) {
      if (chunk->prev && !chunk->prev->used) {
        to_merge.push_front(chunk->memptr);
      }
    }
    if (!to_merge.empty())
      isFreeBlockAvailble = merge_chunks(to_merge, false, size);
  }

  return isFreeBlockAvailble;
}

void CoalescedStringentPooling::pool_free_chunk(void* ptr) const {
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
  bytes_in_use -= chunk->size;
}

} // namespace pool_allocator
} // namespace synapse_helpers
