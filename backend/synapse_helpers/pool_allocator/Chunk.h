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
#pragma once
#include "absl/container/flat_hash_set.h"
#include "backend/synapse_helpers/device_types.h"

using stream_set = absl::flat_hash_set<synapse_helpers::hpuStream_t>;

namespace synapse_helpers {
namespace pool_allocator {

struct Chunk {
  uint64_t size;
  uint64_t extra_space;
  bool used;
  Chunk* prev;
  Chunk* next;
  uint64_t memptr;
  uint64_t bin_index;
  uint64_t freed_counter;
  synapse_helpers::hpuStream_t stream; // allocation stream
  bool associated_to_stream;
  stream_set stream_uses; // streams on which the block was used
  int event_count; // number of outstanding HPU events

  Chunk(size_t sz)
      : size(sz),
        used(false),
        prev(nullptr),
        next(nullptr),
        memptr(0),
        bin_index(-1),
        freed_counter(0),
        stream(0),
        associated_to_stream(0),
        event_count(0) {}
  Chunk(size_t sz, hpuStream_t s)
      : size(sz),
        used(false),
        prev(nullptr),
        next(nullptr),
        memptr(0),
        bin_index(-1),
        freed_counter(0),
        stream(s),
        associated_to_stream(0),
        event_count(0) {}
  Chunk()
      : size(0),
        extra_space(0),
        used(false),
        prev(nullptr),
        next(nullptr),
        memptr(0),
        bin_index(-1),
        freed_counter(0),
        stream(0),
        associated_to_stream(0),
        event_count(0) {}
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
