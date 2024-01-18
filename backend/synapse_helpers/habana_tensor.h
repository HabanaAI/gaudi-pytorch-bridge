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

#include "habana_helpers/logging.h"

#include "backend/synapse_helpers/layout_utils.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "backend/synapse_helpers/value_or_ref.h"

inline std::ostream& operator<<(std::ostream& out, const synTensorType& t) {
  switch (t) {
    case DATA_TENSOR:
      out << "DATA_TENSOR";
      break;
    case DATA_TENSOR_DYNAMIC:
      out << "DATA_TENSOR_DYNAMIC";
      break;
    case SHAPE_TENSOR:
      out << "SHAPE_TENSOR";
      break;
    case HOST_TO_DEVICE_TENSOR:
      out << "HOST_TO_DEVICE_TENSOR";
      break;
    case DEVICE_SHAPE_TENSOR:
      out << "DEVICE_SHAPE_TENSOR";
      break;
    case TENSOR_TYPE_MAX:
    default:
      HABANA_ASSERT(false);
  }
  return out;
}

namespace synapse_helpers {

const uint64_t INVALID_SYN_TENSOR_ID = std::numeric_limits<uint64_t>::max();

class memory_section {
 public:
  // explicit c'tor that holds valid synSectionHandle
  explicit memory_section(synSectionHandle section)
      : memory_section_{section} {}
  // c'tor that creates synSectionHandle
  memory_section(uint64_t memory_attributes, synGraphHandle graph);
  ~memory_section() {
    if (memory_section_) {
      // To Do - make a provision to destroy at the end of use case for shape
      // agnostic as sections would not be destoryed by graph destroy
      if (!is_sa_on_) {
        synSectionDestroy(memory_section_);
      }
    }
  }
  memory_section(const memory_section&) = delete;
  memory_section& operator=(const memory_section&) = delete;

  operator synSectionHandle() {
    return memory_section_;
  }

  synSectionHandle GetSectionHandle() {
    return memory_section_;
  }

  void set_sa_on(const bool flag) {
    is_sa_on_ = flag;
  }

  bool is_sa_on() const {
    return is_sa_on_;
  }

 private:
  synSectionHandle memory_section_;
  // flag to recognize if this section is part of shape agnostic cached graph
  bool is_sa_on_{false};
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
    using dimension_size_t = int64_t;
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
        return value >= rhs.value;
      }

      unsigned value{};
    };

    using internal_storage = std::array<dimension_size_t, HABANA_DIM_MAX>;
    explicit shape_t(
        dimension_count_t rank = dimension_count_t{0},
        std::initializer_list<dimension_size_t> dimensions = {})
        : rank_(rank) {
      HABANA_ASSERT(
          dimensions.size() == 0 ||
              (rank.value == dimensions.size() && rank.value <= HABANA_DIM_MAX),
          "Wrong number of dimensions specified, dim size ",
          dimensions.size(),
          " rank size ",
          rank.value);

      unsigned index = 0;
      for (const auto& dim : dimensions) {
        dims_[index++] = dim;
      }
      for (; index < HABANA_DIM_MAX; index++) {
        dims_[index] = 1;
      }
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

  static tensor create_placeholder(
      const std::vector<int64_t>& pt_shape,
      const std::vector<int64_t>& pt_stride);

  static tensor create_placeholder(
      synDeviceId device_id,
      const std::vector<int64_t>& pt_shape,
      const std::vector<int64_t>& pt_stride,
      synDataType data_type,
      bool persistent = false,
      const std::string& suffix = std::string(),
      synTensorType tensor_type = DATA_TENSOR,
      bool tensor_id_inc_flag = true);

  static tensor create_placeholder(
      synDeviceId device_id,
      const std::vector<int64_t>& pt_shape,
      const std::vector<int64_t>& pt_stride,
      bool persistent = false,
      const std::string& suffix = std::string(),
      synTensorType tensor_type = DATA_TENSOR,
      bool tensor_id_inc_flag = true);

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
  const shape_t& stride() const {
    return stride_.max();
  }

  const synapse_helpers::layouts::MemoryPermutation& permutation() const {
    return permutation_;
  }

  bool set_dont_allow_permute(bool allow) {
    return dont_allow_permute_ = allow;
  }

  bool is_dont_allow_permute() const {
    return dont_allow_permute_;
  }

  void set_identity_permutation() {
    for (size_t i = 0; i < permutation_.size(); ++i) {
      permutation_[i] = i;
    }
    set_permutation();
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
  bool is_external() const {
    return is_external_;
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

  bool is_device_shape_tensor() const {
    return tensor_type_ == DEVICE_SHAPE_TENSOR;
  }

  bool is_host_to_device_tensor() const {
    return tensor_type_ == HOST_TO_DEVICE_TENSOR;
  }

  void set_intermediate_shape_tensor() {
    is_intermediate_shape_ = true;
  }

  bool is_intermediate_shape_tensor() const {
    return is_intermediate_shape_;
  }

  synTensorType tensor_type() const {
    return tensor_type_;
  };

  std::vector<int64_t> pt_shape() const {
    return pt_shape_;
  }
  std::vector<int64_t> pt_strides() const {
    return pt_strides_;
  }

  uint64_t id() const {
    return tensor_id_;
  }

  void set_pt_info(
      const std::vector<int64_t>& pt_shape,
      const std::vector<int64_t>& pt_stride) {
    pt_shape_ = pt_shape;
    pt_strides_ = pt_stride;
  }

  void set_host_ptr(void* host_ptr) {
    host_ptr_ = host_ptr;
  }

  void set_inference_range(float min, float max) {
    have_quantization_data_ = true;
    dynamic_range_.min = min;
    dynamic_range_.max = max;
  }

  void set_quant_params(unsigned exp_bias, float scale) {
    has_quant_params_ = true;
    quant_params_.expBias = exp_bias;
    quant_params_.scale = scale;
  }

  synFpQuantParam get_quant_params() const {
    return quant_params_;
  }

  bool has_quant_params() const {
    return has_quant_params_;
  }

  uint64_t get_host_ptr() const {
    return reinterpret_cast<uint64_t>(host_ptr_);
  }

  void set_host_ptr_size(uint64_t host_ptr_size) {
    host_ptr_size_ = host_ptr_size;
  }

  uint64_t get_host_ptr_size() const {
    return host_ptr_size_;
  }

  bool is_shape_agnostic() const {
    return is_shape_agnostic_;
  }

  void set_shape_agnostic_on(bool flag) {
    is_shape_agnostic_ = flag;
  }

  friend std::ostream& operator<<(std::ostream& out, const tensor& rhs);

  std::string DebugString(int indent = 0) const;

  static bool generate_placeholder() {
    return generate_placeholder_;
  }

  static void set_generate_placeholder(bool f) {
    generate_placeholder_ = f;
  }
  synapse_error_o set_permutation();

 private:
  tensor(
      synDeviceId device_id,
      synDataType data_type,
      uint64_t total_size_bytes,
      shape_t shape,
      shape_t stride,
      std::string tensor_name,
      uint64_t tensor_id,
      synGraphHandle graph,
      bool is_persistent = false,
      bool is_external = false,
      shared_memory_section memory_section = nullptr,
      bool is_const = false,
      bool is_const_section = false,
      void* host_ptr = nullptr,
      const uint64_t host_ptr_size = 0,
      const uint64_t offset = 0,
      synTensorType tensor_type = DATA_TENSOR,
      synapse_helpers::layouts::MemoryPermutation memory_permutation = {});

  tensor(
      synDeviceId device_id,
      synDataType data_type,
      uint64_t total_size_bytes,
      const dynamic_shape_t& shape,
      const dynamic_shape_t& stride,
      std::string tensor_name,
      uint64_t tensor_id,
      synGraphHandle graph,
      bool is_persistent = false,
      bool is_external = false,
      shared_memory_section memory_section = nullptr,
      bool is_const = false,
      bool is_const_section = false,
      void* host_ptr = nullptr,
      const uint64_t host_ptr_size = 0,
      const uint64_t offset = 0,
      synTensorType tensor_type = DATA_TENSOR,
      synapse_helpers::layouts::MemoryPermutation memory_permutation = {});

  void set_placeholder() {
    placeholder_ = true;
  }
  synapse_error_o create_old_synapi();
  synapse_error_o create();
  void cleanup();
  synapse_error_o set_quantization_dynamic_range();
  synapse_error_o set_quantization_params();

  std::string tensor_name_;
  uint64_t tensor_id_{INVALID_SYN_TENSOR_ID};
  synDeviceId device_id_;
  synDataType data_type_;
  synQuantDynamicRange dynamic_range_{0, 0};
  synFpQuantParam quant_params_{};
  // TODO: total size can be counted basing on type and dimensions
  uint64_t total_size_bytes_;
  dynamic_shape_t shape_;
  dynamic_shape_t stride_;
  synTensor tensor_{nullptr};
  bool placeholder_{false};
  bool have_quantization_data_{false};
  bool has_quant_params_{false};
  bool is_persistent_{false};
  bool is_external_{false};
  bool is_intermediate_shape_{false};

  shared_memory_section memory_section_{nullptr};
  synGraphHandle graph_{nullptr};
  bool is_const_{false};
  bool is_const_section_{false};
  void* host_ptr_{nullptr};
  uint64_t host_ptr_size_{0};
  const uint64_t offset_{0};
  synTensorType tensor_type_{DATA_TENSOR};
  /*
   * This pt_shape_ tensor is used to store the shape
   * as we receive from pytorch tensor. This value is only
   * used for propogating the shape value when creating
   * placeholder tensor for shape inference. The format
   * of shape stored in pt_shape_ does not match with the
   * format stored in 'shape_'.
   */
  std::vector<int64_t> pt_shape_;
  std::vector<int64_t> pt_strides_;
  static bool generate_placeholder_;

  // permutaion representation for passing strided weight tensor to Synapse
  synapse_helpers::layouts::MemoryPermutation permutation_;
  bool dont_allow_permute_ = false;

  bool is_shape_agnostic_ = false;
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

inline std::ostream& operator<<(
    std::ostream& out,
    const tensor::shape_t& dimensions) {
  out << "syn_dimensions=(";
  std::string delim = "";
  for (auto i = 0u; i < dimensions.rank().value; ++i) {
    out << delim << dimensions[i];
    delim = ", ";
  }
  return out << ")";
}

inline std::ostream& operator<<(
    std::ostream& out,
    const tensor::dynamic_shape_t& d) {
  if (d.min() == d.max()) {
    return out << d.min();
  }
  return out << "min : " << d.min() << ", max : " << d.max();
}

template <
    typename Integer,
    typename = std::enable_if_t<std::is_integral<Integer>::value>>
inline std::ostream& operator<<(
    std::ostream& out,
    const std::vector<Integer>& d) {
  out << "[";
  for (size_t i = 0; i < d.size(); ++i) {
    out << (i > 0 ? ", " : "") << (unsigned)d[i];
  }
  return out << "]";
}

inline std::ostream& operator<<(std::ostream& out, const tensor& tensor) {
  return out << tensor.DebugString(/*indent=*/2);
}

using tensor_or_ref = value_or_ref<tensor>;
using synapse_tensor_ref = std::reference_wrapper<synapse_helpers::tensor>;
} // namespace synapse_helpers

CREATE_OSTREAM_FORMATTER(synapse_helpers::tensor::shape_t);
CREATE_OSTREAM_FORMATTER(synapse_helpers::tensor);
