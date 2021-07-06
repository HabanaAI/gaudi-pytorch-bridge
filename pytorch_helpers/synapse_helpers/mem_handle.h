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
#include <ostream>
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

  bool is_valid() {
    return id_ != 0;
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

} // namespace synapse_helpers
