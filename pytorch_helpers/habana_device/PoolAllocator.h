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
#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <synapse_api_types.h>
#include <synapse_helpers/device.h>
#include <synapse_helpers/habana_tensor.h>

namespace at {
namespace habana {

enum PoolStrategyType {
    strategy_none = 0,
    strategy_bump,
    strategy_dynamic,
};

// [Fix Me:] need to have the pool size to accomodate one
// complete model for static pooling
#define DEFAULT_POOL_SIZE  1000000000 //1GBit
#define POOLING_TYPE  strategy_bump
#define DEFAULT_ALIGNMENT 128


#define allocateHostMemory  new
#define freeHostMemory  delete

class PoolingStrategy
{
 public:
    virtual ~PoolingStrategy() {}
    virtual void* pool_create(synDeviceId deviceID, size_t size) const = 0;
    virtual void pool_destroy(void *p) const = 0;
    virtual void* pool_alloc_chunk(void *p, size_t size) const = 0;
    virtual void pool_free_chunk(void *p) const = 0;
};

class SubAllocator
{
 private:
    PoolingStrategy *strategy_;
 public:
    SubAllocator(PoolingStrategy *strategy = nullptr) : strategy_(strategy) {
	}

    ~SubAllocator() {
        delete this->strategy_;
    }

    void set_strategy(PoolingStrategy *strategy) {
        delete this->strategy_;
        this->strategy_ = strategy;
    }

    void* pool_create(synDeviceId deviceID, size_t size) const {
        return this->strategy_->pool_create(deviceID, size);
    }

    void pool_destroy(void* p) const {
        return this->strategy_->pool_destroy(p);
    }

    void* pool_alloc_chunk(void *p, size_t size) const {
        return this->strategy_->pool_alloc_chunk(p, size);
    }

    void pool_free_chunk(void *p) const {
        return this->strategy_->pool_free_chunk(p);
    }

};

/// bump pooling ///

struct Poolchunk {
    size_t      size;
    bool        used;
    Poolchunk   *next;
    uint64_t    memptr;
};

struct simple_pool_t {
    char        *next;
    char        *end;
    Poolchunk   *_start;
    Poolchunk   *_top;
    uint64_t    memptr;
};

class StaticPooling : public PoolingStrategy {
 private:
    mutable synDeviceId pool_id;
    mutable uint64_t max_pool_size;
    mutable uint64_t block_count;
    mutable uint64_t allocted_block_size;
    mutable uint64_t free_chunks;
    mutable uint64_t free_chunks_size;
    mutable simple_pool_t *prealloc_pool;
    void* reuse_chunks(void *p, size_t size) const;
    void* get_free_chunk(void *p, size_t size) const;
    void print_pool_stats() const;
    mutable std::mutex sp_mutex;

 public:
    StaticPooling();
    void* pool_create(synDeviceId deviceID, size_t size) const override;
    void pool_destroy(void *p) const override;
    void* pool_alloc_chunk(void *p, size_t size) const override;
    void pool_free_chunk(void *p) const override;
};

/// Variable length pooling using equal fit block ///

struct Block {
    size_t     size;
    bool       used;
    Block      *next;
    uint64_t   memptr;
};

class DynamicPooling : public PoolingStrategy {
 private:
    mutable synDeviceId pool_id;
    mutable Block *pool_start;
    mutable Block *top;
    Block *retrieveBlock(void *data) const;
    Block *requestNewBlock(size_t size) const;
    Block *equalFit(size_t size) const;
    Block *findBlock(size_t size) const;
    void *allocBlock(size_t size) const;
    void freeBlock(void *data) const;
    void freeBlocks(Block* base_block) const;
    mutable std::mutex vp_mutex;

 public:
    DynamicPooling();
    void* pool_create(synDeviceId deviceID, size_t size) const override;
    void pool_destroy(void *p) const override;
    void* pool_alloc_chunk(void *p, size_t size) const override;
    void pool_free_chunk(void *p) const override;
};

} // namespace habana
} // namespace at
