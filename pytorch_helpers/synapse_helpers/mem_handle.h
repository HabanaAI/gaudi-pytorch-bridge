/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 *******************************************************************************/
#pragma once

#include <cstdint>
#include <deque>
#include <ostream>
#include <queue>
#include "synapse_helpers/device_types.h"

namespace synapse_helpers {

class mem_handle {
 public:
  using id_t = uint32_t;
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
  static void ensure_fits_ptr(const mem_handle&);
  static constexpr uint64_t offset_bits = 32;
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
    PtrSize() = default;
    PtrSize(size_t size) : size_(size){};
    PtrSize(void* ptr, size_t size) : ptr_(ptr), size_(size){};
  };

  HandlesMap();

  mem_handle::id_t Insert(int size);
  PtrSize GetPtrSize(mem_handle::id_t id) const;
  void SetPtrSize(mem_handle::id_t id, PtrSize ptr_size);
  void Erase(mem_handle::id_t id);
  void MarkMemoryFixed(mem_handle::id_t id);

  struct MemoryRecord {
    mem_handle::id_t id_;
    bool fixed_;
    PtrSize ptr_size_;
  };
  // Iterator has been added for Defragmentator
  struct Iterator {
    Iterator(const HandlesMap& handle_set, mem_handle::id_t id)
        : handle_set_(handle_set), id_(id) {}

    const MemoryRecord operator*() const {
      return {
          .id_ = id_,
          .fixed_ = handle_set_.handle_[id_].fixed_,
          .ptr_size_ = handle_set_.handle_[id_].ptr_size_};
    }

    Iterator& operator++();
    Iterator operator++(int);

    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.id_ == b.id_;
    };
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return a.id_ != b.id_;
    };

   private:
    const HandlesMap& handle_set_;
    mem_handle::id_t id_;
  };
  Iterator begin() const {
    return Iterator(*this, 1);
  };
  Iterator end() const {
    return Iterator(*this, handle_.size());
  };

 private:
  struct Record {
    PtrSize ptr_size_{};
    bool fixed_ = false;
    bool active_ = false;
    Record() = default;
    Record(size_t size) : ptr_size_(size), active_(true) {}
  };
  void CheckId(mem_handle::id_t id) const;

  std::deque<Record> handle_;
  std::queue<mem_handle::id_t> free_handles_;
};

} // namespace synapse_helpers
