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
#include <synapse_common_types.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/synapse_helpers/synapse_error.h"

namespace synapse_helpers {
class device;

class host_memory {
 public:
  explicit host_memory(device& device);
  ~host_memory(); //= default;
  host_memory(const host_memory&) = delete;
  host_memory& operator=(const host_memory&) = delete;
  host_memory(host_memory&&) = delete;
  host_memory& operator=(host_memory&&) = delete;
  synStatus malloc(void** ptr, size_t size);
  synStatus free(void* ptr);
  void dropCache();
  bool is_host_memory(void* ptr);

 private:
  struct BlockSize {
    size_t size; // allocation size
    void* ptr; // host memory pointer

    BlockSize(size_t size, void* ptr = nullptr) : size(size), ptr(ptr) {}
  };

  struct Block : public BlockSize {
    bool allocated; // true if the block is currently allocated

    Block(size_t size, void* ptr, bool allocated)
        : BlockSize(size, ptr), allocated(allocated) {}
  };

  static bool BlockComparator(const BlockSize& a, const BlockSize& b) {
    // sort by size, break ties with pointer
    if (a.size != b.size) {
      return a.size < b.size;
    }
    return (uintptr_t)a.ptr < (uintptr_t)b.ptr;
  }
  using Comparison = bool (*)(const BlockSize&, const BlockSize&);

  // lock around all operations
  std::mutex mutex_;

  device& device_;
  // pointers that are ready to be allocated
  std::set<BlockSize, Comparison> available_;

  // blocks by pointer
  std::unordered_map<void*, Block> blocks;
};
} // namespace synapse_helpers
