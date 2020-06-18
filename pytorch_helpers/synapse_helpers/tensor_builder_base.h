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

#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/habana_tensor.h"

// next step todo:
// - mutual exclusion
//   - static asserts for resetting with conflicting value ?
// - cleanup of emplace functions family

namespace synapse_helpers {

namespace detail {
std::string generate_name();
uint64_t size_bytes_from_shape(
    const tensor::shape_t& shape,
    synDataType dataType);
} // namespace detail

template <typename ConcreteBuilder>
class tensor_builder_base {
 public:
  explicit tensor_builder_base(const tensor& tensor) {
    with_shape(tensor.shape());
    with_data_type(tensor.type());
  }

  explicit tensor_builder_base(
      const tensor::shape_t& shape,
      synDataType data_type = synDataType::syn_type_float)
      : data_type_{data_type} {
    with_shape(shape);
  };

  explicit tensor_builder_base(synDataType data_type) : data_type_{data_type} {}

  ConcreteBuilder& with_data_type(synDataType data_type) {
    data_type_ = data_type;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_shape(const tensor::shape_t& shape) {
    shape_ = shape;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& with_rank_at_least(unsigned required_rank) {
    const auto previous_rank = shape_.rank().value;
    shape_.set_rank(tensor::shape_t::dimension_count_t{
        std::max(required_rank, previous_rank)});

    for (auto i = previous_rank; i < shape_.rank().value; i++) {
      shape_[i] = 1;
    }

    return static_cast<ConcreteBuilder&>(*this);
  }

  // NOLINTNEXTLINE // we're move()'ing, so no const& is needed. TODO remove
  // this line when we switch to tidy-10.
  ConcreteBuilder& with_name(std::string name) {
    tensor_name_ = std::move(name);
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_persistence(const bool is_persistent = true) {
    is_persistent_ = is_persistent;
    return static_cast<ConcreteBuilder&>(*this);
  }

  ConcreteBuilder& mark_const(
      const bool is_const = true,
      void* host_ptr = nullptr) {
    is_const_ = is_const;
    host_ptr_ = host_ptr;
    return static_cast<ConcreteBuilder&>(*this);
  }

  // NOLINTNEXTLINE // we're move()'ing, so no const& is needed. TODO remove
  // this line when we switch to tidy-10.
  ConcreteBuilder& with_memory_section(shared_memory_section memory_section) {
    memory_section_ = std::move(memory_section);
    return static_cast<ConcreteBuilder&>(*this);
  }

  synapse_error_v<tensor> build(device& syn_device, synGraphHandle graph)
      const {
    auto t = tensor(
        syn_device.id(),
        data_type_,
        total_size_bytes(),
        shape_,
        tensor_name_,
        graph,
        is_persistent_,
        memory_section_,
        is_const_,
        host_ptr_);

    auto create_result{t.create()};

    if (create_result.has_value()) {
      return create_result.value();
    } else {
      return {std::move(t)};
    }
  }

 protected:
  tensor_builder_base() = default;

 private:
  tensor::shape_t shape_{};
  synDataType data_type_{};
  std::string tensor_name_ = generate_name();
  bool is_persistent_{false};
  bool is_const_{false};
  shared_memory_section memory_section_{nullptr};
  void* host_ptr_{nullptr};

  uint64_t total_size_bytes() const {
    return detail::size_bytes_from_shape(shape_, data_type_);
  }

  static std::string generate_name() {
    return detail::generate_name();
  }
};

class generic_tensor_builder
    : public tensor_builder_base<generic_tensor_builder> {
 public:
  using tensor_builder_base::tensor_builder_base;
};

} // namespace synapse_helpers
