/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <synapse_common_types.h>
#include <iterator>
#include <sstream>
#include <utility>

#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"

namespace synapse_helpers {
host_memory::host_memory(device& device)
    : mutex_{}, device_{device}, available_(BlockComparator) {}

host_memory::~host_memory() {
  std::lock_guard<std::mutex> lock(mutex_);
  dropCache();
}

synStatus host_memory::malloc(void** ptr, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (device_.HostMemoryCacheEnabled_()) {
    /* search for the smallest block which can hold this allocation */
    BlockSize search_key(size);
    auto it = available_.lower_bound(search_key);
    if (it != available_.end()) {
      Block& block = blocks.at(it->ptr);
      block.allocated = true;
      *ptr = block.ptr;
      available_.erase(it);
      return synSuccess;
    }
  }

  *ptr = nullptr;
  /* allocate a new block if no cached allocation is found */
  auto err = synHostMalloc(device_.id(), size, 0, ptr);
  /* release the cache and retry malloc if the error is OOM */
  if (err == synOutOfHostMemory) {
    PT_SYNHELPER_WARN(
        "SynHostMalloc Failed OOM, Retrying by dropping cache.", err);
    dropCache();
    err = synHostMalloc(device_.id(), size, 0, ptr);
  }
  if (err != synSuccess) {
    return err;
  }

  blocks.insert({*ptr, Block(size, *ptr, true)});
  return synSuccess;
}

static void free_memory(device* d, void* ptr) {
  auto err = synHostFree(d->id(), ptr, 0);
  if (err != synSuccess) {
    // FIXME since the destuctor are not called correctly from device
    // call to synHostFree fails.
    PT_SYNHELPER_DEBUG("SynHostFree Failed.", err);
  }
}

synStatus host_memory::free(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!ptr) {
    return synSuccess;
  }

  auto it = blocks.find(ptr);
  HABANA_ASSERT(it != blocks.end());

  Block& block = it->second;
  HABANA_ASSERT(block.allocated);

  block.allocated = false;
  if (device_.HostMemoryCacheEnabled_()) {
    available_.insert(block);
  } else {
    free_memory(&device_, ptr);
    blocks.erase(it);
  }
  return synSuccess;
}

void host_memory::dropCache() {
  /* clear list of available blocks */
  available_.clear();

  /* free and erase non-allocated blocks */
  for (auto it = blocks.begin(); it != blocks.end();) {
    Block& block = it->second;
    if (!block.allocated) {
      free_memory(&device_, block.ptr);
      it = blocks.erase(it);
    } else {
      ++it;
    }
  }
}
bool host_memory::is_host_memory(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ptr) {
    return false;
  }

  auto it = blocks.find(ptr);
  if (it == blocks.end()) {
    return false;
  } else {
    Block& block = it->second;
    if (block.allocated)
      return true;
    else
      return false;
  }
}
} // namespace synapse_helpers
