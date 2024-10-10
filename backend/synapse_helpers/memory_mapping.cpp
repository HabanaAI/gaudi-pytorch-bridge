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
#include "backend/synapse_helpers/memory_mapping.h"

#include <absl/memory/memory.h>
#include <synapse_api.h>

#include <iterator>
#include <sstream>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "backend/synapse_helpers/device.h"
#include "habana_helpers/logging.h"

namespace synapse_helpers {

memory_mapper::fixed_size_entries::fixed_size_entries(
    mapping_size_t size,
    device& dev)
    : size_(size), device_{dev} {}

memory_mapper::fixed_size_entries::~fixed_size_entries() {
  if (unmap_all() != synStatus::synSuccess) {
    // At this point, we can just log that we failed
    PT_SYNHELPER_WARN(
        "Failed to unmap host memory buffers with total_bytes=", size_);
  }
}

memory_mapper::acquired_entry memory_mapper::fixed_size_entries::acquire() {
  lock_t lock(entries_lock_);
  auto it = std::find_if(
      mapped_entries_.begin(),
      mapped_entries_.end(),
      [](const mapped_entry& elem) { return !elem.in_use; });
  if (it == mapped_entries_.end()) {
    auto allocated_buf = absl::make_unique<uint8_t[]>(size_); // NOLINT
    auto status = synHostMap(device_.id(), size_, allocated_buf.get());
    if (status != synStatus::synSuccess) {
      return {size_, 0, nullptr, status};
    } else {
      auto* ptr = allocated_buf.get();
      mapped_entries_.emplace_back(true, std::move(allocated_buf));
      return {size_, mapped_entries_.size() - 1, ptr, synStatus::synSuccess};
    }
  } else {
    // alloc new
    it->in_use = true;
    return {
        size_,
        static_cast<std::size_t>(std::distance(mapped_entries_.begin(), it)),
        it->buf.get(),
        synStatus::synSuccess};
  }
}

void memory_mapper::fixed_size_entries::release(std::size_t idx) {
  lock_t lock(entries_lock_);
  if (idx >= mapped_entries_.size()) {
    PT_SYNHELPER_WARN(
        "Warning: Entry not found in the cache of mapped buffers!");
    return;
  }
  if (!mapped_entries_[idx].in_use) {
    PT_SYNHELPER_WARN("Warning: Entry already released before!");
    return;
  }
  mapped_entries_[idx].in_use = false;
}

synStatus memory_mapper::fixed_size_entries::unmap_all() {
  lock_t lock(entries_lock_);
  synStatus retStatus = synStatus::synSuccess;
  for (const auto& entry : mapped_entries_) {
    if (!entry.in_use) {
      auto status = synHostUnmap(device_.id(), entry.buf.get());
      if (status != synStatus::synSuccess) {
        retStatus = status;
      }
    }
  }
  // clean up any buffers not in use anymore
  mapped_entries_.erase(
      std::remove_if(
          mapped_entries_.begin(),
          mapped_entries_.end(),
          [](const mapped_entry& entry) { return !entry.in_use; }),
      mapped_entries_.end());
  return retStatus;
}

memory_mapper::memory_mapper(device& device)
    : device_{device}, mapped_locations_{} {}

memory_mapper::acquired_entry memory_mapper::map(mapping_size_t size) {
  lock_t lock(locations_access_);
  auto it = mapped_locations_.find(size);
  if (it == mapped_locations_.end()) {
    auto result = mapped_locations_.emplace(
        size, std::make_shared<fixed_size_entries>(size, device_));
    if (!result.second) {
      return {size, 0, nullptr, synStatus::synFail};
    }
    it = result.first;
  }
  auto elem = it->second;
  lock.unlock();
  return elem->acquire();
}

void memory_mapper::unmap(acquired_entry entry) {
  lock_t lock(locations_access_);
  auto it = mapped_locations_.find(entry.acquired_size);
  if (it == mapped_locations_.end()) {
    PT_SYNHELPER_WARN(
        "Warning: Table for entries with size: ",
        entry.acquired_size,
        " released before!");
    return;
  }
  auto mapped_location_for_size = it->second;
  lock.unlock();
  mapped_location_for_size->release(entry.idx);
}

synStatus memory_mapper::drop_cache() {
  lock_t lock(locations_access_);
  synStatus retStatus = synStatus::synSuccess;
  for (const auto& entry : mapped_locations_) {
    auto status = entry.second->unmap_all();
    if (status != synStatus::synSuccess) {
      retStatus = status;
    }
  }
  return retStatus;
}

} // namespace synapse_helpers
