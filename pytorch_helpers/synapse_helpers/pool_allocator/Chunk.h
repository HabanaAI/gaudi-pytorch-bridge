/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
namespace synapse_helpers {
namespace pool_allocator {

struct Chunk {
  uint64_t size;
  uint64_t extra_space;
  bool used;
  Chunk* prev;
  Chunk* next;
  uint64_t memptr;

  Chunk(size_t sz)
      : size(sz), used(false), prev(nullptr), next(nullptr), memptr(0) {}
  Chunk()
      : size(0),
        extra_space(0),
        used(false),
        prev(nullptr),
        next(nullptr),
        memptr(0) {}
};

struct simple_coalesced_pool_t {
  uint64_t next;
  uint64_t end;
  Chunk* start;
  Chunk* top;
  uint64_t memptr;
  uint64_t basememptr;
};
} // namespace pool_allocator
} // namespace synapse_helpers
