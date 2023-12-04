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

#include "util.h"

namespace synapse_helpers {
static std::array<HandleBucketInfo, _END> bucketInfo = {
    {{small_offset_bits,
      (total_bits - small_offset_bits),
      (1ULL << small_offset_bits),
      (1ULL << (total_bits - small_offset_bits))},
     {medium_offset_bits,
      (total_bits - medium_offset_bits),
      (1ULL << medium_offset_bits),
      (1ULL << (total_bits - medium_offset_bits))},
     {big_offset_bits,
      (total_bits - big_offset_bits),
      (1ULL << big_offset_bits),
      (1ULL << (total_bits - big_offset_bits))}}};

/**
 * Function takes mem_handle and converts it so it can be passed
 * around as pointer. This is done because pytorch requires to
 * get the pointer to memory on the device.
 * Offset of the handle is stored in lower bits. This way newly formed
 * pointer conforms to the pointer arithmetic as long as the
 * offsets are not too big.
 */
device_ptr mem_handle::reinterpret_to_pointer(const mem_handle& h) {
  uint64_t id = h.id_;
  bucket_type type = (bucket_type)(id >> total_bits);
  if (type >= _END)
    PT_SYNHELPER_FATAL("Wrong  Bucket Type", type);

  uint64_t offset = h.offset_;
  uint64_t offset_bits = bucketInfo[type].offset_bits;
  uint64_t inside_bucket_id = id & ((1ULL << total_bits) - 1);
  ensure_fits_ptr(type, inside_bucket_id, offset);
  uint64_t combined = id << offset_bits;
  combined |= offset;

  combined =
      create_memhandle_from_bucket_index_and_handle_index(type, combined);
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

  bucket_type type = (bucket_type)(ptr >> total_bits);
  if (type > _END)
    PT_SYNHELPER_FATAL("Wrong  Bucket Type", type);

  uint64_t offset_bits = bucketInfo[type].offset_bits;
  uint64_t mask = (1ULL << total_bits) - 1;
  uint64_t inside_val = mask & ptr;
  uint64_t id = (inside_val & ((1ULL << total_bits) - 1)) >> offset_bits;
  uint64_t offset = inside_val & ((1ULL << offset_bits) - 1);
  id = create_memhandle_from_bucket_index_and_handle_index(type, id);

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

void mem_handle::ensure_fits_ptr(
    bucket_type type,
    uint64_t id,
    uint64_t offset) {
  if (id > bucketInfo[type].max_handles) {
    PT_SYNHELPER_FATAL("Wrong mem_handle id ", id);
  }

  if (offset > bucketInfo[type].max_offsets) {
    PT_SYNHELPER_FATAL("Wrong mem_handle offset", offset);
  }
}

HandlesMap::HandlesMap() {
  handles_[BUCKET_TYPE_SMALL].emplace_back(Record{});
  handles_[BUCKET_TYPE_MEDIUM].emplace_back(Record{});
  handles_[BUCKET_TYPE_BIG].emplace_back(Record{});
}

bucket_type HandlesMap::getBucketIndexForGivenTensorSize(size_t size) {
  for (int bucket_index = 0; bucket_index < (int)bucketInfo.size();
       bucket_index++) {
    if (bucketInfo[bucket_index].max_offsets > size) {
      return (bucket_type)bucket_index;
    }
  }
  PT_SYNHELPER_FATAL("Huge Size not supported");
  return BUCKET_TYPE_SMALL;
}

void HandlesMap::check_id_overflow(
    mem_handle::id_t id,
    size_t size,
    uint64_t offset_bits) {
  uint64_t id_to_check = id;
  auto limit = (id_to_check << offset_bits) + size;
  if ((limit >> offset_bits) != id) {
    PT_SYNHELPER_FATAL("Handle overflow occurred");
  }
}

mem_handle::id_t HandlesMap::Insert(size_t size, hpuStream_t stream) {
  bucket_type bucket_index = getBucketIndexForGivenTensorSize(size);
  const uint64_t offset_bits = bucketInfo[bucket_index].offset_bits;
  if (!free_handles_[bucket_index].empty()) {
    uint64_t id = free_handles_[bucket_index].front();
    check_id_overflow(id, size, offset_bits);
    free_handles_[bucket_index].pop();
    handles_[bucket_index][id] = Record(size, stream);
    return create_memhandle_from_bucket_index_and_handle_index(
        bucket_index, id);
  } else {
    if (handles_[bucket_index].size() > bucketInfo[bucket_index].max_handles) {
      PT_SYNHELPER_FATAL("All possible device memory handles has been used");
      return mem_handle::invalid_handle;
    }
    handles_[bucket_index].emplace_back(size, stream);
    auto id_inside_bucket = handles_[bucket_index].size() - 1;
    check_id_overflow(id_inside_bucket, size, offset_bits);
    return create_memhandle_from_bucket_index_and_handle_index(
        bucket_index, id_inside_bucket);
  }
}

HandlesMap::PtrSize HandlesMap::GetPtrSize(mem_handle::id_t id) const {
  bucket_type bucket_index = get_bucket_index(id);
  uint64_t handle_index = get_handle_index(id);
  CheckId(handle_index, bucket_index);
  return handles_[bucket_index][handle_index].ptr_size_;
}

void HandlesMap::SetPtrSize(mem_handle::id_t id, HandlesMap::PtrSize ptr_size) {
  bucket_type bucket_index = get_bucket_index(id);
  uint64_t handle_index = get_handle_index(id);
  CheckId(handle_index, bucket_index);
  handles_[bucket_index][handle_index].ptr_size_ = ptr_size;
}

void HandlesMap::Erase(mem_handle::id_t id) {
  bucket_type bucket_index = get_bucket_index(id);
  uint64_t handle_index = get_handle_index(id);
  CheckId(handle_index, bucket_index);
  handles_[bucket_index][handle_index].active_ = false;
  handles_[bucket_index][handle_index].ptr_size_.size_ = 0;
  handles_[bucket_index][handle_index].ptr_size_.ptr_ = nullptr;
  free_handles_[bucket_index].push(handle_index);
}

void HandlesMap::MarkMemoryFixed(mem_handle::id_t id) {
  bucket_type bucket_index = get_bucket_index(id);
  uint64_t handle_index = get_handle_index(id);
  CheckId(handle_index, bucket_index);
  handles_[bucket_index][handle_index].fixed_ = true;
}

void HandlesMap::CheckId(uint64_t handle_index, bucket_type index) const {
  if (!mem_handle::is_valid(handle_index) ||
      handle_index >= handles_[index].size() ||
      !handles_[index][handle_index].active_) {
    PT_SYNHELPER_FATAL("Handle doesn't exist");
  }
}

size_t HandlesMap::size() const {
  size_t bucket_size = 0;
  for (int i = 0; i < (int)bucketInfo.size(); i++) {
    bucket_size += handles_[i].size();
  }
  return bucket_size;
}

HandlesMap::Iterator& HandlesMap::Iterator::operator++() {
  if (handleIndex_ < handle_set_.handles_[bucketIndex_].size() - 1) {
    ++handleIndex_;
    while (handleIndex_ < (handle_set_.handles_[bucketIndex_].size() - 1) &&
           !handle_set_.handles_[bucketIndex_][handleIndex_].active_) {
      ++handleIndex_;
    }
  } else {
    do {
      ++bucketIndex_;
    } while (bucketIndex_ < handle_set_.handles_.size() &&
             handle_set_.handles_[bucketIndex_].empty());

    handleIndex_ = 0;
  }
  return *this;
}

HandlesMap::Iterator HandlesMap::Iterator::operator++(int) {
  Iterator tmp = *this;
  ++(*this);
  return tmp;
}

bool HandlesMap::checkIdIsReset(mem_handle::id_t id) {
  bucket_type bucket_index = get_bucket_index(id);
  uint64_t handle_index = get_handle_index(id);
  CheckId(handle_index, bucket_index);
  bool reset = handles_[bucket_index][handle_index].reset_;
  if (handles_[bucket_index][handle_index].active_ && reset) {
    handles_[bucket_index][handle_index].reset_ = false;
  }
  return reset;
}

void HandlesMap::ResetHandlesMap(mem_handle::id_t id) {
  bucket_type bucketIndex = get_bucket_index(id);
  uint64_t handle_index = get_handle_index(id);
  CheckId(handle_index, bucketIndex);
  if (handles_[bucketIndex][handle_index].active_) {
    handles_[bucketIndex][handle_index].reset_ = true;
    handles_[bucketIndex][handle_index].ptr_size_.size_ = 0;
    handles_[bucketIndex][handle_index].ptr_size_.ptr_ = nullptr;
  }
}
} // namespace synapse_helpers
