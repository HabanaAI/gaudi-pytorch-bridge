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
#pragma once

#include <absl/container/flat_hash_map.h>
#include <synapse_common_types.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace synapse_helpers {
class device;

using mapping_size_t = uint64_t;

class memory_mapper {
 private:
  struct acquired_entry;

 public:
  explicit memory_mapper(device& device);
  ~memory_mapper() = default;
  memory_mapper(const memory_mapper&) = delete;
  memory_mapper& operator=(const memory_mapper&) = delete;
  memory_mapper(memory_mapper&&) = delete;
  memory_mapper& operator=(memory_mapper&&) = delete;

  memory_mapper::acquired_entry map(mapping_size_t size);
  void unmap(acquired_entry entry);
  // drop all cached mapped buffers
  synStatus drop_cache();

  friend device;

 private:
  using lock_t = std::unique_lock<std::mutex>;

  struct mapped_entry {
    mapped_entry(bool in_use, std::unique_ptr<uint8_t[]> buf)
        : in_use{in_use}, buf{std::move(buf)} {} // NOLINT
    bool in_use = false;
    std::unique_ptr<uint8_t[]> buf; // NOLINT
  };

  struct acquired_entry {
    mapping_size_t acquired_size;
    size_t idx;
    uint8_t* ptr;
    synStatus status;
  };

  class fixed_size_entries {
   public:
    fixed_size_entries(mapping_size_t size, device& dev);
    ~fixed_size_entries();
    acquired_entry acquire();
    // input index should be the one returned from acquire() call
    void release(std::size_t idx);
    synStatus unmap_all();

   private:
    mapping_size_t size_;
    device& device_;
    std::mutex entries_lock_;
    std::vector<mapped_entry> mapped_entries_;
  };

  device& device_;

  absl::flat_hash_map<mapping_size_t, std::shared_ptr<fixed_size_entries>>
      mapped_locations_;
  std::mutex locations_access_;
};

} // namespace synapse_helpers
