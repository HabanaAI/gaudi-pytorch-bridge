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
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "backend/synapse_helpers/synapse_error.h"
#include "habana_helpers/logging.h"

// next step todo:
// - mutual exclusion
//   - static asserts for resetting with conflicting value ?
// - cleanup of emplace functions family

namespace synapse_helpers {

tensor::shape_t to_shape_t(
    const std::vector<int64_t>& shape,
    bool reverse = true);

tensor::shape_t to_shape_tensor_stride_t(const int64_t& stride_rank);

tensor::shape_t to_stride_t(
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& shape,
    synDataType data_type,
    bool reverse = true);

namespace detail {
class tensor_name_generator {
 public:
  static std::string get_next_tensor_name(
      const std::string& suffix = std::string());
  static std::string generate(
      const std::string& suffix = std::string(),
      bool tensor_id_inc_flag = true);
  static void set_tensor_id(uint64_t id);
  static uint64_t get_tensor_id();
  static void to_netron_syntax(std::string& name);

  // to make unit testing possible
  static void reset();

 private:
  static thread_local uint64_t syn_tensor_id;
};

uint64_t size_bytes_from_shape(
    const tensor::shape_t& shape,
    synDataType dataType);
} // namespace detail

namespace graph_builder {
class graph_build_context;
}
template <typename ConcreteBuilder>
class tensor_builder_base {
  friend class graph_builder::graph_build_context;

 public:
  explicit tensor_builder_base(const tensor& tensor) {
    with_shape(tensor.shape());
    with_stride(tensor.stride());
    with_data_type(tensor.type());
  }

  explicit tensor_builder_base(
      const tensor::shape_t& shape,
      synDataType data_type = synDataType::syn_type_float)
      : data_type_{data_type} {
    with_shape(shape);
  };

  explicit tensor_builder_base(
      const tensor::shape_t& shape,
      const tensor::shape_t& stride,
      synDataType data_type = synDataType::syn_type_float)
      : data_type_{data_type} {
    with_shape(shape);
    with_stride(stride);
  };

  explicit tensor_builder_base(synDataType data_type) : data_type_{data_type} {}

  ConcreteBuilder& with_data_type(synDataType data_type) {
    data_type_ = data_type;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_shape(const tensor::shape_t& shape) {
    shape_ = tensor::dynamic_shape_t{shape, shape};
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_stride(const tensor::shape_t& stride) {
    stride_ = tensor::dynamic_shape_t{stride, stride};
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_permutation(
      const synapse_helpers::layouts::MemoryPermutation& permutation) {
    permutation_ = permutation;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_dont_allow_permutation(const bool allow) {
    dont_allow_permutation_ = allow;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_is_shape_agnostic_on(const bool flag) {
    is_shape_agnostic_on_ = flag;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_inference_range(const float min, const float max) {
    have_quantization_data = true;
    inference_min = min;
    inference_max = max;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_quant_params(
      const unsigned exp_bias,
      const float scale = 1.0) {
    has_quant_params_ = true;
    exp_bias_ = exp_bias;
    scale_ = scale;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_dynamic_shape(
      const tensor::dynamic_shape_t& dynamic_shape) {
    HABANA_ASSERT(dynamic_shape.max().rank() == dynamic_shape.min().rank());
    if (is_const_) {
      // Const tensors must always be created as static (with max size)
      return static_cast<ConcreteBuilder&>(*this);
    }
    shape_ = dynamic_shape;
    if (tensor_type_ == DATA_TENSOR) {
      tensor_type_ = DATA_TENSOR_DYNAMIC;
    }
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_dynamic_stride(
      const tensor::dynamic_shape_t& dynamic_stride) {
    HABANA_ASSERT(dynamic_stride.max().rank() == dynamic_stride.min().rank());
    if (is_const_) {
      // Const tensors must always be created as static (with max size)
      return static_cast<ConcreteBuilder&>(*this);
    }
    stride_ = dynamic_stride;
    if ((tensor_type_ == DATA_TENSOR) && (shape_.min() != shape_.max())) {
      tensor_type_ = DATA_TENSOR_DYNAMIC;
    }
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_shape_and_stride(
      const tensor::shape_t& shape,
      const tensor::shape_t& stride) {
    // With C++17, this can move to std::variant
    shape_ = tensor::dynamic_shape_t{shape, shape};
    stride_ = tensor::dynamic_shape_t{stride, stride};
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_dynamic_shape_and_stride(
      const tensor::dynamic_shape_t& dynamic_shape,
      const tensor::dynamic_shape_t& dynamic_stride) {
    HABANA_ASSERT(dynamic_shape.max().rank() == dynamic_shape.min().rank());
    HABANA_ASSERT(dynamic_stride.max().rank() == dynamic_stride.min().rank());
    HABANA_ASSERT(dynamic_shape.max().rank() == dynamic_stride.max().rank());
    shape_ = dynamic_shape;
    stride_ = dynamic_stride;
    if ((tensor_type_ == DATA_TENSOR) && (shape_.min() != shape_.max())) {
      tensor_type_ = DATA_TENSOR_DYNAMIC;
    }
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_rank_at_least(unsigned required_rank) {
    const auto previous_rank = shape_.rank().value;
    shape_.set_rank(tensor::shape_t::dimension_count_t{
        std::max(required_rank, previous_rank)});

    for (auto i = previous_rank; i < shape_.rank().value; i++) {
      shape_.set_dim(i, 1);
    }

    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& use_suffix(const std::string& suffix) {
    suffix_ = suffix;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_persistence(const bool is_persistent = true) {
    is_persistent_ = is_persistent;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_external(const bool is_external = true) {
    is_external_ = is_external;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& set_offset(const uint64_t offset = 0) {
    offset_ = offset;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_const(
      const bool is_const = true,
      void* host_ptr = nullptr,
      const uint64_t host_ptr_size = 0) {
    is_const_ = is_const;
    host_ptr_ = host_ptr;
    host_ptr_size_ = host_ptr_size;
    HABANA_ASSERT(tensor_type_ == DATA_TENSOR);
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_const_section(
      const bool is_const_section = true,
      void* host_ptr = nullptr) {
    is_const_section_ = is_const_section;
    if (is_const_section_) {
      host_ptr_ = host_ptr;
    }
    HABANA_ASSERT(tensor_type_ == DATA_TENSOR);
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_shape_tensor() {
    tensor_type_ = SHAPE_TENSOR;
    data_type_ = syn_type_uint32;
    is_persistent_ = true;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_device_shape_tensor() {
    tensor_type_ = DEVICE_SHAPE_TENSOR;
    data_type_ = syn_type_uint32;
    // device shape tensors are still 5-element (SYN_MAX_TENSOR_DIM)
    return with_shape(tensor::shape_t(1_D, {SYN_MAX_TENSOR_DIM}));
  }

  ConcreteBuilder& mark_host_to_device_tensor(void* host_ptr) {
    HABANA_ASSERT(host_ptr != nullptr);
    tensor_type_ = HOST_TO_DEVICE_TENSOR;
    host_ptr_ = host_ptr;
    data_type_ = syn_type_int32;
    is_persistent_ = true;
    return static_cast<ConcreteBuilder&>(*this);
  }

  // NOLINTNEXTLINE // we're move()'ing, so no const& is needed. TODO remove
  // this line when we switch to tidy-10.
  ConcreteBuilder& with_memory_section(shared_memory_section memory_section) {
    memory_section_ = std::move(memory_section);
    return static_cast<ConcreteBuilder&>(*this);
  }

  /*ConcreteBuilder& mark_variable(bool mark = true) {
    if (mark) {
      section_type_ = kSectionTypeTfVariables;
      is_persistent_ = true;
    };
    return static_cast<ConcreteBuilder&>(*this);
  }*/

  tensor::dynamic_shape_t& shape() {
    return shape_;
  };

  tensor::dynamic_shape_t& stride() {
    return stride_;
  };

  synapse_error_v<tensor> build(device& syn_device, synGraphHandle graph)
      const {
    if (error_invalid_shape_) {
      return {
          synapse_error{"Unsupported tensor shape", synStatus::synUnsupported}};
    }
    if (error_invalid_dtype_) {
      return {
          synapse_error{"Unsupported tensor dtype", synStatus::synUnsupported}};
    }
    auto tensor_id = detail::tensor_name_generator::get_tensor_id();
    std::string tensor_name = generate_name();
    auto t = tensor(
        syn_device.id(),
        data_type_,
        total_size_bytes(),
        shape_,
        stride_,
        tensor_name,
        tensor_id,
        graph,
        is_persistent_,
        is_external_,
        memory_section_,
        is_const_,
        is_const_section_,
        host_ptr_,
        host_ptr_size_,
        offset_,
        tensor_type_,
        permutation_);
    if (have_quantization_data)
      t.set_inference_range(inference_min, inference_max);
    if (has_quant_params_)
      t.set_quant_params(exp_bias_, scale_);
    t.set_dont_allow_permute(dont_allow_permutation_);
    t.set_shape_agnostic_on(is_shape_agnostic_on_);

    auto create_result{t.create()};

    if (create_result.has_value()) {
      return create_result.value();
    } else {
      return {std::move(t)};
    }
  }

 protected:
  tensor_builder_base() = default;
  void set_error_invalid_shape() {
    error_invalid_shape_ = true;
  }
  void set_error_invalid_dtype() {
    error_invalid_dtype_ = true;
  }

 private:
  tensor::dynamic_shape_t shape_{};
  tensor::dynamic_shape_t stride_{};
  synDataType data_type_{};
  std::string suffix_ = "";
  bool is_persistent_{false};
  bool is_external_{false};
  bool is_const_{false};
  bool is_const_section_{false};
  bool have_quantization_data{false};
  float inference_min = 0, inference_max = 0;
  bool has_quant_params_{false};
  unsigned exp_bias_ = 0;
  float scale_ = 0.0;
  shared_memory_section memory_section_{nullptr};
  void* host_ptr_{nullptr};
  uint64_t host_ptr_size_{0};
  uint64_t offset_{0};
  synTensorType tensor_type_{DATA_TENSOR};
  bool error_invalid_shape_{false};
  bool error_invalid_dtype_{false};
  synapse_helpers::layouts::MemoryPermutation permutation_;
  bool dont_allow_permutation_{false};
  bool is_shape_agnostic_on_{false};

  uint64_t total_size_bytes() const {
    return detail::size_bytes_from_shape(shape_.max(), data_type_);
  }

  std::string generate_name() const {
    return detail::tensor_name_generator::generate(suffix_);
  }
};

class generic_tensor_builder
    : public tensor_builder_base<generic_tensor_builder> {
 public:
  using tensor_builder_base::tensor_builder_base;
};

} // namespace synapse_helpers
