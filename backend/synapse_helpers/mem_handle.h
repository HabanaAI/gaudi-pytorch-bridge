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

#include <cstdint>
#include <deque>
#include <iostream>
#include <ostream>
#include <queue>
#include "backend/synapse_helpers/device_types.h"
#include "habana_helpers/logging.h"

namespace synapse_helpers {
enum bucket_type {
  _BEGIN = 0,
  BUCKET_TYPE_SMALL = 0,
  BUCKET_TYPE_MEDIUM = 1,
  BUCKET_TYPE_BIG = 2,
  _END = 3
};

// Handle - Total of 64 bit, 2 bits are used  for Bucket
// type and reset for handle id and offset bit
static const uint64_t total_bits = 62;
static const uint64_t small_offset_bits = 22;
static const uint64_t medium_offset_bits = 32;
static const uint64_t big_offset_bits = 42;

struct HandleBucketInfo {
  const uint64_t offset_bits;
  const uint64_t handle_bits;
  uint64_t max_offsets;
  uint64_t max_handles;
};

#define create_memhandle_from_bucket_index_and_handle_index(A, B) \
  (((uint64_t)A << total_bits) | B)
#define get_max_offset(A) (1ULL << (uint64_t)A)
#define get_bucket_index(A) ((bucket_type)((uint64_t)A >> total_bits))
#define get_handle_index(A) (A & ((1ULL << total_bits) - 1))

class mem_handle {
 public:
  using id_t = uint64_t;
  using offset_t = uint64_t;

  bool operator==(const mem_handle& rhs) const {
    return id_ == rhs.id_ && offset_ == rhs.offset_;
  }
  bool operator!=(const mem_handle& rhs) const {
    return !operator==(rhs);
  }

  static mem_handle reinterpret_from_pointer(device_ptr ptr);
  static device_ptr reinterpret_to_pointer(const mem_handle& h);

  static constexpr id_t invalid_handle = 0;
  bool is_valid() {
    return id_ != invalid_handle;
  }
  static bool is_valid(id_t id) {
    return id != invalid_handle;
  }

  id_t id() const {
    return id_;
  }
  offset_t offset() const {
    return offset_;
  }
  mem_handle unoffseted() const {
    return mem_handle(id_);
  }

  explicit mem_handle(id_t id) : id_(id) {}

 private:
  mem_handle(id_t id, offset_t offset) : id_(id), offset_(offset) {}
  static void ensure_fits_ptr(bucket_type type, uint64_t id, uint64_t offset);
  id_t id_ = 0;
  offset_t offset_ = 0;

  friend std::ostream& operator<<(std::ostream& os, const mem_handle& h);
  friend struct std::hash<mem_handle>;
};

inline std::ostream& operator<<(std::ostream& os, const mem_handle& h) {
  os << "{ mem_handle@" << h.id_ << "[" << h.offset_ << "] }";
  return os;
}

class HandlesMap {
 public:
  struct PtrSize {
    void* ptr_ = nullptr;
    size_t size_ = 0;
    hpuStream_t stream_ = 0;
    PtrSize() = default;
    PtrSize(size_t size, hpuStream_t stream) : size_(size), stream_(stream){};
    PtrSize(void* ptr, size_t size, hpuStream_t stream)
        : ptr_(ptr), size_(size), stream_(stream){};
  };

  HandlesMap();

  mem_handle::id_t Insert(size_t size, hpuStream_t stream = 0);
  void check_id_overflow(mem_handle::id_t id, size_t size, uint64_t offset_bit);
  PtrSize GetPtrSize(mem_handle::id_t id) const;
  void SetPtrSize(mem_handle::id_t id, PtrSize ptr_size);
  void Erase(mem_handle::id_t id);
  void MarkMemoryFixed(mem_handle::id_t id);
  bool checkIdIsReset(mem_handle::id_t id);
  void ResetHandlesMap(mem_handle::id_t id);
  bucket_type getBucketIndexForGivenTensorSize(size_t size);
  struct MemoryRecord {
    mem_handle::id_t id_;
    bool fixed_;
    PtrSize ptr_size_;
  };
  // Iterator has been added for Defragmentator
  struct Iterator {
    Iterator(
        const HandlesMap& handle_set,
        size_t bucketIndex,
        size_t handleIndex)
        : handle_set_(handle_set),
          bucketIndex_(bucketIndex),
          handleIndex_(handleIndex) {
      id_ = create_memhandle_from_bucket_index_and_handle_index(
          (bucket_type)bucketIndex_, handleIndex_);
    }

    const MemoryRecord operator*() const {
      return {
          .id_ = create_memhandle_from_bucket_index_and_handle_index(
              (bucket_type)bucketIndex_, handleIndex_),
          .fixed_ = handle_set_.handles_[bucketIndex_][handleIndex_].fixed_,
          .ptr_size_ =
              handle_set_.handles_[bucketIndex_][handleIndex_].ptr_size_};
    }

    Iterator& operator++();
    Iterator operator++(int);

    friend bool operator==(const Iterator& a, const Iterator& b) {
      return (a.bucketIndex_ == b.bucketIndex_) &&
          (a.handleIndex_ == b.handleIndex_);
    };
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return (a.bucketIndex_ != b.bucketIndex_) &&
          (a.handleIndex_ != b.handleIndex_);
    };

   private:
    const HandlesMap& handle_set_;
    size_t bucketIndex_ = 0;
    size_t handleIndex_ = 0;
    mem_handle::id_t id_;
  };
  Iterator begin() const {
    return Iterator(*this, 0, 0);
  };
  Iterator end() const {
    return Iterator(*this, this->handles_.size(), size());
  };

  std::size_t size() const;

 private:
  struct Record {
    PtrSize ptr_size_{};
    bool fixed_ = false;
    bool active_ = false;
    bool reset_ = false;
    Record() = default;
    Record(size_t size, hpuStream_t stream)
        : ptr_size_(size, stream), active_(true) {}
  };
  void CheckId(uint64_t handle_index, bucket_type index) const;

  std::array<std::deque<Record>, _END> handles_;
  std::array<std::queue<mem_handle::id_t>, _END> free_handles_;
};

} // namespace synapse_helpers

CREATE_OSTREAM_FORMATTER(synapse_helpers::mem_handle);
