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
