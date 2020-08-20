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

#include "../HPUAllocator.h"
#include "../HPUCheck.h"
#include "../HPUGuardImpl.h"
#include "../hpu_cached_devices.h"
#include "PoolAllocator.h"
#include "CoalescedPoolAllocator.h"
#include "utils.h"
#include <habana_helpers/logging.h>

#define DEFRAGMENT_TH(arg) std::ceil(0.9*(arg))
//#define DEFRAGMENT_ON_REUSE

namespace at {
namespace habana {
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

void * StaticCoalescedPooling::pool_create(synDeviceId deviceID, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    size = block_align(size);
    pool_id = deviceID;
    uint64_t free_mem, total_mem;
    auto status = synDeviceGetMemoryInfo(deviceID, &free_mem, &total_mem);
    if (synStatus::synSuccess != status) {
        PT_DEVICE_FATAL("POOL:: Cannot obtain device memory size. Status: ", status);
    }
    if (size > free_mem) {
        PT_DEVICE_DEBUG("POOL:: requested size is more than avaiable memory");
        //setting the pool size to 90% of available memory in case of failure
        size = 0.9 * free_mem;
        PT_DEVICE_DEBUG("POOL:: set new pool size : ", size);
        std::cout << "POOL:: cannot set requested pool size, free mem :: " << free_mem << " size used for pool :: " << size << "\n";
    }
    max_pool_size = size;

    auto p = new simple_coalesced_pool_t();
    if (!p) {
        PT_DEVICE_FATAL("POOL:: Cannot obtain pool memory");
        return nullptr;
    }

    status = synDeviceMalloc(pool_id, size, 0, 0, &p->basememptr);
    if (synStatus::synSuccess != status) {
        delete(p);
        PT_DEVICE_FATAL("POOL:: Cannot obtain device memory size. Status: ", status);
        return nullptr;
    }

    //0x80 bytes left for future use - header maintence in device memory instead of host
    p->memptr = p->basememptr + 0x80;
    p->next = p->memptr;
    p->end = p->basememptr + size;
    p->start = nullptr;
    p->top = p->start;
    PT_DEVICE_DEBUG("POOL:: static coalesced pool created");
    print_device_memory_stats(pool_id);
    prealloc_pool = p;

    //StaticCoalescedPooling::print_pool_stats();
    PT_DEVICE_DEBUG("POOL:: Pool Created :: base host :: ", p ," base ptr :: ", p->basememptr, \
        " memptr :: ", p->memptr, " prealloc_pool :: ", prealloc_pool->memptr , \
        " end :: ", p->end, " next :: ", p->next);

    return p;
}

void StaticCoalescedPooling::pool_destroy(void *ptr) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    simple_coalesced_pool_t *s_pool = prealloc_pool;

    StaticCoalescedPooling::print_pool_stats();

    if ((s_pool) && (chunk_count != 0)) {
        PT_DEVICE_DEBUG("POOL:: warning -- active chunks !!");
        PT_DEVICE_DEBUG("POOL:: total active chunks :: ", chunk_count);
    }

    if (s_pool) {
        if (!get_device_deallocation()) {

            if (nullptr != (void*)s_pool->basememptr) {
                uint64_t ptr_address{reinterpret_cast<uint64_t>(s_pool->basememptr)};
                auto status{synDeviceFree(pool_id, ptr_address, 0)};
                if (status) {
                    //TORCH_HABANA_CHECK(status, "synDeviceFree failed");
                    PT_DEVICE_DEBUG("POOL:: synDeviceFree failed :: ", status);
                    set_device_deallocation(true);
                }
            }
        }
        set_device_deallocation(false);

        s_pool->basememptr = 0;
        std::list<Chunk*>::iterator it;
        for (it = pool_list.begin(); it != pool_list.end(); ++it) {
            delete(*it);
        }
        pool_list.clear();
        delete(s_pool);
        s_pool = nullptr;
        PT_DEVICE_DEBUG("POOL:: static coalesced pool destroyed");
        print_device_memory_stats(pool_id);
    }
}

static uint64_t pool_available(simple_coalesced_pool_t *p) {
    return p->end - p->next;
}

void StaticCoalescedPooling::print_pool_stats() const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    static const std::string occupancy_mask = "[+++]";
    static const std::string free_mask     = "[00000]";
    static std::stringstream   pool_status;
    static int occupied_chunks = 0;
    static int total_chunks = 0;
    static uint64_t occupied_size = 0;
    static uint64_t total_size = 0;
    static int total_extra_spaced_chunks = 0;
    static uint64_t total_exta_size = 0;
    pool_status.str("");
    pool_status.clear();

    for (auto& chunk : pool_list) {
        total_chunks++;
        total_size+=chunk->size;
        if (chunk->extra_space) {
            total_extra_spaced_chunks++;
            total_exta_size+=chunk->extra_space;
        }
        if (chunk->used) {
            occupied_chunks++;
            occupied_size+=chunk->size;
            pool_status << occupancy_mask;
        } else if (!chunk->used && (chunk->size !=0)) {
            pool_status << free_mask;
            free_chunks ++;
            free_chunks_size+=chunk->size;
            if (chunk->prev && !chunk->prev->used) {
                PT_DEVICE_DEBUG ("POOL:: can be merged :: chunk :: ", chunk->memptr, " with prev :: " , chunk->prev->memptr);
            }
            if (chunk->next && !chunk->next->used) {
                PT_DEVICE_DEBUG("POOL:: can be merged :: chunk :: ", chunk->memptr, " with next :: ", chunk->next->memptr);
            }
        }
    }
    PT_DEVICE_DEBUG("POOL:: total_chunks in the pool :: ", total_chunks);
    PT_DEVICE_DEBUG("POOL:: total_size in the pool :: ", total_size);
    PT_DEVICE_DEBUG("POOL:: occupied_chunks in the pool :: ", occupied_chunks);
    PT_DEVICE_DEBUG("POOL:: occupied_chunks size in the pool :: ", occupied_size);
    PT_DEVICE_DEBUG("POOL:: free chunks in the pool :: ", free_chunks);
    PT_DEVICE_DEBUG("POOL:: free chunks size in the pool :: ", free_chunks_size);
    PT_DEVICE_DEBUG("POOL:: total_extra_spaced_chunks in the pool  :: ", total_extra_spaced_chunks);
    PT_DEVICE_DEBUG("POOL:: total_extra_size in the pool chunks :: ", total_exta_size);
    PT_DEVICE_DEBUG("POOL::{}", pool_status.str());
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
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    // skip the chunk if
    // 1. already in use
    // 2. chunk size cannot accomodate requested size
    // 3. requested size is not 90% of the chunk size
    // this is to avoid small sized requests consuming bigger free chunks
    if (chunk->used || (chunk->size < size_req) || (size_req < DEFRAGMENT_TH(chunk->size))) {
         return true;
    }
    return false;
}

Chunk * StaticCoalescedPooling::get_any_available_free_chunk(uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    std::list<Chunk*>::iterator it;
    for (it = pool_list.begin(); it != pool_list.end(); ++it){
        auto chunk = *it;
        if(chunk->used || size > chunk->size) {
            continue;
        }
        PT_DEVICE_DEBUG("POOL:: Return bigger chunk :: ", chunk, " chunk size :: ", chunk->size, " requested size :: ", size);
        return chunk;
    }
    PT_DEVICE_DEBUG("POOL:: no bigger chunks !!");
    return nullptr;
}

void * StaticCoalescedPooling::get_free_chunk(void *ptr, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    std::list<Chunk*>::iterator it;
    for (it = pool_list.begin(); it != pool_list.end(); ++it){
        if (skip_chunk(*it, size)) {
            continue;
        }
        return *it;
    }
    return nullptr;
}

Chunk * StaticCoalescedPooling::defragment_on_reuse(void *ptr, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    simple_coalesced_pool_t *p = (simple_coalesced_pool_t *)ptr;
    auto curr_size = p->end - p->next;
    // do not defragment until pool touches 75% capacity
    if (curr_size >= (0.75*max_pool_size)) {
        PT_DEVICE_DEBUG("POOL:: more than 75% pool available, not defragmenting, \
            current size :: ", curr_size, " pool size :: " , max_pool_size );
        return nullptr;
    }
    return try_defragmenting(ptr, size);
}

Chunk * StaticCoalescedPooling::try_defragmenting(void *ptr, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    simple_coalesced_pool_t *p = (simple_coalesced_pool_t *)ptr;
    auto chunk = p->start;

    //try defragmenting the pool
    //we defragment only for the requested size as there is no
    //pre-allocated bin for different block sizes
    auto isFreeBlockAvailble = pool_defragment(size);
    auto free_chunk = (Chunk*)get_free_chunk(chunk, size);
    if (free_chunk == nullptr) {
        if (isFreeBlockAvailble) {
            PT_DEVICE_DEBUG("POOL:: Free blocks available -- block split, p->end :: ",p->end, \
                " p->next :: ",p->next," max_pool_size :: ", max_pool_size);
            isFreeBlockAvailble = false;
        }
        PT_DEVICE_DEBUG("POOL:: no more reusable chunk after defragment: extend pool !!");
        return nullptr;
    }
    PT_DEVICE_DEBUG("POOL:: reusing chunk after defragment:: ", free_chunk->memptr, " req size :: " ,\
        size, " chunk size :: ", free_chunk->size);
    free_chunk->used = true;
    return free_chunk;
}

Chunk * StaticCoalescedPooling::reuse_chunks(void *ptr, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    simple_coalesced_pool_t *p = (simple_coalesced_pool_t *)ptr;
    auto chunk = p->start;
    auto free_chunk = (Chunk*)get_free_chunk(chunk, size);
    if (free_chunk == nullptr) {
        #ifdef DEFRAGMENT_ON_REUSE
            return defragment_on_reuse(ptr, size);
        #else
            PT_DEVICE_DEBUG("POOL:: no more reusable chunk: extend pool !!");
            return nullptr;
        #endif
    }
    PT_DEVICE_DEBUG("POOL:: reusing chunk :: ", free_chunk->memptr, " req size :: " ,\
        size, " chunk size :: ", free_chunk->size);
    free_chunk->used = true;
    return free_chunk;
}

Chunk * StaticCoalescedPooling::try_block_splitting( uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    Chunk *big_chunk = nullptr;
    big_chunk =  get_any_available_free_chunk(size);
    if (big_chunk) {
        auto split_chunk = try_splitting_chunks(big_chunk, size);
        if (split_chunk) {
            PT_DEVICE_DEBUG("POOL:: bigger chunk split :: big_chunk :: ", big_chunk, \
                " big_chunk size :: ", big_chunk->size, " split chunk :: ", \
                split_chunk, " split chunk size :: ", split_chunk->size);
            return split_chunk;
        }
    }
    return nullptr;
}

void * StaticCoalescedPooling::pool_alloc_chunk(void *ptr, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    size = block_align(size);
    simple_coalesced_pool_t *p = (simple_coalesced_pool_t *)prealloc_pool;
    if (prealloc_pool != p) {
        PT_DEVICE_FATAL("POOL:: alloc unknown pool !!");
    }
    PT_DEVICE_DEBUG("POOL:: pool_alloc_chunk request in pool :: ", p, " for size :: ", size);

    auto old_chunk = reuse_chunks(p, size);
    if (old_chunk) {
        ++chunk_count;
        auto prevptr = old_chunk->prev ? old_chunk->prev->memptr : 0;
        auto nextptr = old_chunk->next ? old_chunk->next->memptr : 0;
        //extra space available in blocks after split/coalasce
        old_chunk->extra_space = old_chunk->size - size;
        PT_DEVICE_DEBUG("POOL:: pool_alloc_chunk allocated reuse chunk :: base:: ", old_chunk, \
            " chunk memptr ::", old_chunk->memptr, " prev :: ", prevptr, " next :: ", nextptr,  \
            " requested size :: ", size, " chunk size :: ", old_chunk->size, \
            " extra space :: ", old_chunk->extra_space);
        return (void*)old_chunk->memptr;
    }

    if (pool_available(p) < size) {
        //TBD: implement better algorithms
        auto defrag_chunk = try_defragmenting(p, size);
        if (defrag_chunk) {
            ++chunk_count;
            return (void*)defrag_chunk->memptr;
        }
        auto split_chunk = try_block_splitting(size);
        if (split_chunk) {
            return (void*)split_chunk->memptr;
        }
        print_device_memory_stats(pool_id);
        print_pool_stats();
        PT_DEVICE_FATAL("POOL:: pool exhausted !! for size :: ", size);
    }

    //create a chunk
    Chunk* chunk = new Chunk();
    if (!chunk) {
        PT_DEVICE_FATAL("POOL:: Cannot create a chunk");
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
    PT_DEVICE_DEBUG("POOL:: pool_alloc_chunk allocated :: base:: ", chunk, \
        " chunk memptr ::", chunk->memptr, " prev :: ", prevptr, \
        " next :: ", nextptr, " requested size :: ", size, " chunk size :: ", chunk->size);
    PT_DEVICE_DEBUG("POOL:: Allocated chunk_count :: ", chunk_count);

    pool_list.push_back(chunk);
    return (void*)chunk->memptr;
}

bool StaticCoalescedPooling::canMergeNextChunk(Chunk *chunk, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
  //return false;
  return (!chunk->used && chunk->next && !chunk->next->used &&
    (DEFRAGMENT_TH(chunk->size + chunk->next->size) >= size));
}

Chunk *StaticCoalescedPooling::mergeNextChunk(Chunk *chunk) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
  auto adj_chunk = chunk->next;
  //check if current chunk and next chunk addresses are contigous
  if (((chunk->memptr + chunk->size) != adj_chunk->memptr ) || (chunk->next->used)) {
      PT_DEVICE_DEBUG("POOL:: Next chunk is not contigous or free ");
      return chunk;
  }

  chunk->next = adj_chunk->next;

  auto prevptr = chunk->prev ? chunk->prev->memptr : 0;
  auto nextptr = chunk->next ? chunk->next->memptr : 0;
  PT_DEVICE_DEBUG("POOL:: ***Merged - next** chunk :: ", chunk->memptr, " prev :: ", \
    prevptr, " next :: ", nextptr, " merged with :: ", adj_chunk->memptr);
  PT_DEVICE_DEBUG("POOL:: ***Merged - next** chunk size :: ", chunk->size + adj_chunk->size, \
    " old size :: ", chunk->size);

  //assuming device memory is contingous
  chunk->size = chunk->size + adj_chunk->size;

  if (prealloc_pool->top == adj_chunk) {
      //update top
      prealloc_pool->top = chunk;
  }

  adj_chunk->used = false;
  adj_chunk->extra_space = 0;
  adj_chunk->size = 0;
  adj_chunk->memptr = 0;
  adj_chunk->next = nullptr;
  adj_chunk->prev = nullptr;

  return chunk;
}

bool StaticCoalescedPooling::canMergePreviousChunk(Chunk *chunk, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
  //return false;
  return (!chunk->used && chunk->prev && !chunk->prev->used &&
    (DEFRAGMENT_TH(chunk->size + chunk->prev->size) >= size));
}

Chunk *StaticCoalescedPooling::mergePreviousChunk(Chunk *chunk) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
  auto adj_chunk = chunk->prev;
  //check if current chunk and previous chunk addresses are contigous
  if (((adj_chunk->memptr + adj_chunk->size) != chunk->memptr ) || (chunk->prev->used)) {
      PT_DEVICE_DEBUG("POOL:: Previous chunk is not contigous ");
      return chunk;
  }

  if (prealloc_pool->top == chunk) {
      //update top
      prealloc_pool->top = adj_chunk;
  }

  //assuming device memory is contingous
  adj_chunk->next = chunk->next;

  auto prevptr = chunk->prev ? chunk->prev->memptr : 0;
  auto nextptr = chunk->next ? chunk->next->memptr : 0;
  PT_DEVICE_DEBUG("POOL:: ***Merged - previous** chunk :: ", chunk->memptr, " prev :: ", \
    prevptr, " next :: ", nextptr, " merged with :: ", adj_chunk->memptr);
  PT_DEVICE_DEBUG("POOL:: ***Merged - previous** chunk size :: ", chunk->size + adj_chunk->size, \
    " old size :: ", chunk->size);

  adj_chunk->size = chunk->size + adj_chunk->size;

  chunk->used = false;
  chunk->extra_space = 0;
  chunk->size = 0;
  chunk->memptr = 0;
  chunk->next = nullptr;
  chunk->prev = nullptr;

  return chunk;
}

Chunk* StaticCoalescedPooling::try_coalescing_chunks(void *ptr, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    auto chunk = (Chunk*)ptr;
    if (canMergePreviousChunk(chunk, size)) {
        //coalesce adjacent free chunks
        chunk = mergePreviousChunk(chunk);
        return chunk;
    }
    if (canMergeNextChunk(chunk, size)) {
        //coalesce adjacent free chunks
        chunk = mergeNextChunk(chunk);
        return chunk;
    }
    if (chunk->prev && chunk->next) {
        if (!chunk->prev->used && !chunk->next->used) {
            // try merging the missed out chunks
            //chunk = mergePreviousChunk(chunk);
            chunk = mergeNextChunk(chunk);
        }
    }
    return chunk;
}

Chunk* StaticCoalescedPooling::create_chunk(uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    //create a chunk
    Chunk* chunk = new Chunk();
    if (!chunk) {
        PT_DEVICE_FATAL("POOL:: Cannot create a chunk");
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

Chunk* StaticCoalescedPooling::try_splitting_chunks(void *ptr, uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    auto chunk = (Chunk*)ptr;
    auto new_chunk = create_chunk(size);
    if (!new_chunk) {
        return chunk;
    }
    auto new_size = chunk->size - size;
    //TBD: ensure memory is contigous
    new_chunk->memptr = chunk->memptr + new_size;
    new_chunk->size = size;
    new_chunk->used = false;
    new_chunk->next = chunk->next;
    new_chunk->prev = chunk;

    if (prealloc_pool->top == chunk) {
        //update top
        prealloc_pool->top = new_chunk;
    }
    chunk->next = new_chunk;
    chunk->size = new_size;

    if ((chunk->memptr + new_size) != new_chunk->memptr) {
        PT_DEVICE_DEBUG("POOL:: split blocks not contigous");
        return chunk;
    }
    pool_list.push_back(new_chunk);
    ++chunk_count;
    return new_chunk;
}

bool StaticCoalescedPooling::pool_defragment(uint64_t size) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    bool isFreeBlockAvailble = false;
    for (auto& chunk : pool_list) {
        if (!chunk->used) {
            // try to merge adjacent free chunks
            chunk = try_coalescing_chunks(chunk, size);
            isFreeBlockAvailble = true;
            if (chunk->size >= size) {
                PT_DEVICE_DEBUG("POOL:: pool defragmentation succeeded for requested size :: ", size, \
                    " chunk->size :: ", chunk->size);
                if (size < DEFRAGMENT_TH(chunk->size)) {
                    PT_DEVICE_DEBUG("POOL:: try block splitting for size :: ", size, " in chunk of chunk->size :: ", \
                        chunk->size);
                    auto newchunk = try_splitting_chunks(chunk, size);
                    PT_DEVICE_DEBUG("POOL:: chunk splitted successfully :: newchunk :: ", newchunk, \
                        " new chunk size :: ", newchunk->size," old chunk :: ", chunk, \
                        " old chunk->size :: ", chunk->size);
                }
                break;
            }
        }
    }
    return isFreeBlockAvailble;
}

void StaticCoalescedPooling::pool_free_chunk(void *ptr) const {
    const std::lock_guard<std::recursive_mutex> lock(sp_mutex);
    //just for debug - figure out allocations outside of pool
    bool allocated_using_pool = false;
    for (auto& chunk : pool_list) {
        if (chunk->memptr == (uint64_t)ptr) {
            chunk->used = false;
            allocated_using_pool = true;
            break;
        }
    }
    if (!allocated_using_pool) {
        PT_DEVICE_DEBUG("POOL: not allocated using pool but freed :: ", ptr);
        uint64_t ptr_address{reinterpret_cast<uint64_t>(ptr)};
        auto status{synDeviceFree(pool_id, ptr_address, 0)};
        TORCH_HABANA_CHECK(status, "synDeviceFree failed");
    } else {
        --chunk_count;
    }
}

} // pool_allocator
} //habana
} //at