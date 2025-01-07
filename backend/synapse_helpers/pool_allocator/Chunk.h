/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
