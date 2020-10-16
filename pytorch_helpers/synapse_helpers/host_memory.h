/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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

#include "synapse_helpers/synapse_error.h"

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

    BlockSize(size_t size, void* ptr = NULL) : size(size), ptr(ptr) {}
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
  typedef bool (*Comparison)(const BlockSize&, const BlockSize&);

  // lock around all operations
  std::mutex mutex_;

  device& device_;
  // pointers that are ready to be allocated
  std::set<BlockSize, Comparison> available_;

  // blocks by pointer
  std::unordered_map<void*, Block> blocks;
};
} // namespace synapse_helpers
