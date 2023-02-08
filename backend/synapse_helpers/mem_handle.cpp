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

#include "backend/synapse_helpers/mem_handle.h"
#include <bitset>
#include <ostream>
#include <type_traits>
#include "habana_helpers/logging.h"

namespace synapse_helpers {

/**
 * Function takes mem_handle and converts it so it can be passed
 * around as pointer. This is done because pytorch requires to
 * get the pointer to memory on the device.
 * Offset of the handle is stored in lower bits. This way newly formed
 * pointer conforms to the pointer arithmetic as long as the
 * offsets are not too big.
 */
device_ptr mem_handle::reinterpret_to_pointer(const mem_handle& h) {
  ensure_fits_ptr(h);

  uint64_t id = h.id_;
  uint64_t offset = h.offset_;
  uint64_t combined = id << offset_bits;
  combined |= offset;

  PT_SYNHELPER_DEBUG(
      "reinterpret_to_pointer h.id::",
      id,
      " offset::",
      offset,
      " ptr::",
      combined);
  return combined;
}

/**
 * Sibling function to the one above. It converts back the
 * pointer from our artificial space into mem_handle object
 */
mem_handle mem_handle::reinterpret_from_pointer(device_ptr ptr) {
  static_assert(
      std::is_same<device_ptr, uint64_t>::value,
      "following code assumes ptr is uint64_t");

  uint64_t offset = ptr & ((1ULL << offset_bits) - 1);
  uint64_t id = ptr >> offset_bits;

  PT_SYNHELPER_DEBUG(
      "reinterpret_from_pointer  h.id::",
      id,
      " offset::",
      offset,
      " Ptr::",
      ptr);
  return mem_handle(id, offset);
}

namespace {
struct mem_handle_runtime_check {
  mem_handle_runtime_check() {
    if (mem_handle::reinterpret_from_pointer(device_nullptr).is_valid()) {
      PT_SYNHELPER_FATAL("device_nullptr should produce invalid mem_handle");
    }
  }
} _;
} // namespace

void mem_handle::ensure_fits_ptr(const mem_handle& h) {
  constexpr static int bits_in_byte = 8;
  if (h.id_ > std::bitset<sizeof(device_ptr) * bits_in_byte - offset_bits>()
                  .set()
                  .to_ullong()) {
    PT_SYNHELPER_FATAL("Wrong mem_handle id ", h.id_);
  }

  if (h.offset_ > std::bitset<offset_bits>().set().to_ullong()) {
    PT_SYNHELPER_FATAL("Wrong mem_handle offset", h.offset_);
  }
}

HandlesMap::HandlesMap() {
  handle_.emplace_back(Record{});
}

void HandlesMap::check_id_overflow(mem_handle::id_t id, size_t size) {
  uint64_t id_to_check = id;
  auto limit = (id_to_check << offset_bits) + size;
  if ((limit >> offset_bits) != id) {
    // TBD for larger handles will be addressed in SW-110816
    PT_SYNHELPER_FATAL("Handle overflow occurred");
  }
}
mem_handle::id_t HandlesMap::Insert(size_t size) {
  if (!free_handles_.empty()) {
    auto id = free_handles_.front();
    check_id_overflow(id, size);
    free_handles_.pop();
    handle_[id] = Record(size);
    PT_SYNHELPER_DEBUG("Reuse handleid ::", id, " Size::", size);
    return id;
  } else {
    constexpr static int bits_in_byte = 8;
    if (handle_.size() >
        std::bitset<sizeof(device_ptr) * bits_in_byte - offset_bits>()
            .set()
            .to_ullong()) {
      // TBD for larger handles will be addressed in SW-110816
      PT_SYNHELPER_FATAL("All possible device memory handles has been used");
      return mem_handle::invalid_handle;
    }
    handle_.emplace_back(size);
    auto id = handle_.size() - 1;
    check_id_overflow(id, size);
    PT_SYNHELPER_DEBUG("Insert new handleid ::", id, " Size::", size);
    return id;
  }
}

HandlesMap::PtrSize HandlesMap::GetPtrSize(mem_handle::id_t id) const {
  CheckId(id);
  return handle_[id].ptr_size_;
}

void HandlesMap::SetPtrSize(mem_handle::id_t id, HandlesMap::PtrSize ptr_size) {
  CheckId(id);
  handle_[id].ptr_size_ = ptr_size;
}

void HandlesMap::Erase(mem_handle::id_t id) {
  CheckId(id);
  handle_[id].active_ = false;
  handle_[id].ptr_size_.size_ = 0;
  handle_[id].ptr_size_.ptr_ = nullptr;
  PT_SYNHELPER_DEBUG("Free handle h.id:", id, " active::", handle_[id].active_);
  free_handles_.push(id);
}

void HandlesMap::MarkMemoryFixed(mem_handle::id_t id) {
  CheckId(id);
  handle_[id].fixed_ = true;
}

void HandlesMap::CheckId(mem_handle::id_t id) const {
  if (!mem_handle::is_valid(id) || id >= handle_.size() ||
      !handle_[id].active_) {
    PT_SYNHELPER_FATAL("Handle doesn't exist");
  }
}

HandlesMap::Iterator& HandlesMap::Iterator::operator++() {
  ++id_;
  while (id_ < handle_set_.handle_.size() &&
         !handle_set_.handle_[id_].active_) {
    ++id_;
  }
  return *this;
}

HandlesMap::Iterator HandlesMap::Iterator::operator++(int) {
  Iterator tmp = *this;
  ++(*this);
  return tmp;
}

void HandlesMap::ResetHandlesMap() {
  handle_.clear();
  handle_.emplace_back(Record{});
  free_handles_ = {};
}
} // namespace synapse_helpers
