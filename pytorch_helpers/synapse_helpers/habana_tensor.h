/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <absl/strings/str_format.h>
#include <synapse_api.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>

#include "synapse_helpers/synapse_error.h"
#include "synapse_helpers/value_or_ref.h"

namespace synapse_helpers {

class memory_section {
 public:
  // explicit c'tor that holds valid synSectionHandle
  explicit memory_section(synSectionHandle section)
      : memory_section_{section} {}
  // c'tor that creates synSectionHandle
  memory_section(uint64_t memory_attributes, synGraphHandle graph);
  ~memory_section() {
    if (memory_section_)
      synSectionDestroy(memory_section_);
  }
  memory_section(const memory_section&) = delete;
  memory_section& operator=(const memory_section&) = delete;

  operator synSectionHandle() {
    return memory_section_;
  }

 private:
  synSectionHandle memory_section_;
};

using shared_memory_section = std::shared_ptr<memory_section>;

class tensor final {
  template <typename>
  friend class tensor_builder_base;

 public:
  tensor() = delete;

  ~tensor();
  tensor(const tensor&) = delete;
  tensor& operator=(const tensor&) = delete;
  tensor(tensor&&) noexcept;
  tensor& operator=(tensor&&) noexcept;

  class shape_t {
   public:
    using dimension_size_t = unsigned;
    struct dimension_count_t {
      explicit dimension_count_t(unsigned arg = 0) : value{arg} {}
      bool operator==(const dimension_count_t& rhs) const {
        return value == rhs.value;
      }
      bool operator!=(const dimension_count_t& rhs) const {
        return value != rhs.value;
      }
      bool operator<=(const dimension_count_t& rhs) const {
        return value <= rhs.value;
      }
      bool operator>=(const dimension_count_t& rhs) const {
        return value <= rhs.value;
      }

      unsigned value{};
    };

    using internal_storage = std::array<dimension_size_t, SYN_MAX_TENSOR_DIM>;
    explicit shape_t(
        dimension_count_t rank = dimension_count_t{0},
        dimension_size_t a = 1,
        dimension_size_t b = 1,
        dimension_size_t c = 1,
        dimension_size_t d = 1,
        dimension_size_t e = 1)
        : dims_{{a, b, c, d, e}} {
      set_rank(rank);
    }

    internal_storage::reference operator[](size_t index) {
      return dims_.at(index);
    }
    internal_storage::const_reference operator[](size_t index) const {
      return dims_.at(index);
    }
    bool operator==(const shape_t& rhs) const noexcept {
      return dims_ == rhs.dims_ && rank_ == rhs.rank_;
    }
    bool operator!=(const shape_t& rhs) const noexcept {
      return dims_ != rhs.dims_ || rank_ != rhs.rank_;
    }

    internal_storage::pointer data() noexcept {
      return dims_.data();
    }
    internal_storage::const_pointer data() const noexcept {
      return dims_.data();
    }

    internal_storage::iterator begin() noexcept {
      return dims_.begin();
    }
    internal_storage::iterator end() noexcept {
      return dims_.end();
    }
    internal_storage::const_iterator begin() const noexcept {
      return dims_.begin();
    }
    internal_storage::const_iterator end() const noexcept {
      return dims_.end();
    }
    internal_storage::const_iterator cbegin() const noexcept {
      return dims_.cbegin();
    }
    internal_storage::const_iterator cend() const noexcept {
      return dims_.cend();
    }

    dimension_count_t rank() const noexcept {
      return dimension_count_t{rank_.value};
    }
    void set_rank(dimension_count_t rank) noexcept;

    std::string debug_string() const;

   private:
    internal_storage dims_;
    dimension_count_t rank_;
  };

  class dynamic_shape_t {
    friend class tensor;

   public:
    explicit dynamic_shape_t(shape_t min = shape_t{}, shape_t max = shape_t{});

    bool operator==(const dynamic_shape_t& rhs) const noexcept {
      return min_ == rhs.min_ && max_ == rhs.max_;
    }
    bool operator!=(const dynamic_shape_t& rhs) const noexcept {
      return min_ != rhs.min_ || max_ != rhs.max_;
    }

    const shape_t& min() const noexcept {
      return min_;
    }
    const shape_t& max() const noexcept {
      return max_;
    }

    void set_dim(size_t index, shape_t::dimension_size_t size) {
      min_[index] = size;
      max_[index] = size;
    }

    void set_dim(
        size_t index,
        shape_t::dimension_size_t min,
        shape_t::dimension_size_t max) {
      min_[index] = min;
      max_[index] = max;
    }

    shape_t::dimension_count_t rank() const noexcept {
      return max_.rank();
    }

    void set_rank(shape_t::dimension_count_t rank) noexcept {
      min_.set_rank(rank);
      max_.set_rank(rank);
    }

   private:
    shape_t min_;
    shape_t max_;
  };

  static tensor create_placeholder(synDeviceId device_id);

  synTensor& get() {
    return tensor_;
  }
  const synTensor& get() const {
    return tensor_;
  }
  const std::string& name() const {
    return tensor_name_;
  }
  uint64_t size_bytes() const {
    return total_size_bytes_;
  }
  uint64_t num_elements() const;
  const shape_t& shape() const {
    return shape_.max();
  }
  synDataType type() const {
    return data_type_;
  }
  synDeviceId device_id() const {
    return device_id_;
  }
  synGraphHandle graph() const {
    return graph_;
  }

  bool is_placeholder() const {
    return placeholder_;
  }
  bool is_persistent() const {
    return is_persistent_;
  }
  bool is_const() const {
    return is_const_;
  }
  shared_memory_section memorysection() const {
    return memory_section_;
  }

  uint64_t get_offset() const {
    return offset_;
  }

  bool has_dynamic_shape() const {
    return shape_.min() != shape_.max();
  }
  const dynamic_shape_t& dynamic_shape() const {
    return shape_;
  }

  bool is_shape_tensor() const {
    return tensor_type_ == SHAPE_TENSOR;
  }
  bool is_input_shape_tensor() const {
    return tensor_type_ == INPUT_DESCRIBING_SHAPE_TENSOR;
  }
  bool is_device_shape_tensor() const {
    return tensor_type_ == DEVICE_SHAPE_TENSOR;
  }
  synTensorType tensor_type() const {
    return tensor_type_;
  };

  friend std::ostream& operator<<(std::ostream& out, const tensor& rhs);

  std::string DebugString() const {
    return absl::StrFormat(
        "Tensor %s at %p, internal=%p%s%s, size=0x%x",
        tensor_name_,
        this,
        tensor_,
        (is_persistent() ? ", persistent" : ", non-persistent"),
        (is_placeholder() ? ", placeholder" : ""),
        total_size_bytes_);
  }

 private:
  tensor(
      synDeviceId device_id,
      synDataType data_type,
      uint64_t total_size_bytes,
      shape_t shape,
      std::string tensor_name,
      synGraphHandle graph,
      bool is_persistent = false,
      shared_memory_section memory_section = nullptr,
      bool is_const = false,
      void* host_ptr = nullptr,
      const uint64_t host_ptr_size = 0,
      const uint64_t offset = 0,
      synTensorType tensor_type = DATA_TENSOR);
  tensor(
      synDeviceId device_id,
      synDataType data_type,
      uint64_t total_size_bytes,
      const dynamic_shape_t& shape,
      std::string tensor_name,
      synGraphHandle graph,
      bool is_persistent = false,
      shared_memory_section memory_section = nullptr,
      bool is_const = false,
      void* host_ptr = nullptr,
      const uint64_t host_ptr_size = 0,
      const uint64_t offset = 0,
      synTensorType tensor_type = DATA_TENSOR);

  void set_placeholder() {
    placeholder_ = true;
  }
  synapse_error_o create_old_synapi();
  synapse_error_o create();
  void cleanup();

  std::string tensor_name_;
  synDeviceId device_id_;
  synDataType data_type_;
  // TODO: total size can be counted basing on type and dimensions
  uint64_t total_size_bytes_;
  dynamic_shape_t shape_;
  synTensor tensor_{nullptr};
  bool placeholder_{false};
  bool is_persistent_{false};

  shared_memory_section memory_section_{nullptr};
  synGraphHandle graph_{nullptr};
  bool is_const_{false};
  void* host_ptr_{nullptr};
  uint64_t host_ptr_size_{0};
  const uint64_t offset_{0};
  synTensorType tensor_type_{DATA_TENSOR};
};

/**
 * @brief Converts number of dimensions to dimension_count_t type.
 *        Allows defining dimensions using integer literals i.e. auto matrix =
 * tensor::shape_t{2_D};
 *
 * @param arg number of dimensions as integer
 * @return number of dimension as dimension_count_t
 */
tensor::shape_t::dimension_count_t operator"" _D(unsigned long long arg);

inline std::ostream& operator<<(std::ostream& out, const tensor& tensor) {
  return out << "Tensor " << tensor.tensor_name_ << " at " << &tensor
             << ", internal=" << tensor.tensor_
             << (tensor.is_persistent() ? ", persistent, "
                                        : ", non-persistent, ")
             << (tensor.is_placeholder() ? "placeholder, " : "")
             << "size=" << tensor.total_size_bytes_;
}

inline std::ostream& operator<<(
    std::ostream& out,
    const tensor::shape_t& dimensions) {
  out << "syn_dimensions=(";
  auto i{dimensions.begin()};
  out << *i;
  for (i++; i != dimensions.end(); i++) {
    out << ", " << *i;
  }
  out << ") rank=(" << dimensions.rank().value << ")";
  return out;
}

using tensor_or_ref = value_or_ref<tensor>;
using synapse_tensor_ref = std::reference_wrapper<synapse_helpers::tensor>;
} // namespace synapse_helpers
