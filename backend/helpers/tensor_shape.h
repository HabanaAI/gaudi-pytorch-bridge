/******************************************************************************
 * Copyright (C) 2020 Habana Labs, Ltd. an Intel Company
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
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <synapse_api_types.h>
#include <iostream>
#include <vector>
#include "habana_helpers/logging_pt.h"

namespace habana_helpers {

class TensorShape {
 public:
  TensorShape() = default;
  TensorShape(const at::IntArrayRef& sizes, at::ScalarType scalar_type);
  void add_dim(int64_t size);
  size_t dims() const {
    return m_dim;
  }
  int64_t dim_size(size_t dim) const {
    TORCH_CHECK(dim < m_dim, "dim idx is out of range");
    return m_sizes[dim];
  }
  int64_t num_elements() const {
    return n_elements;
  }
  std::vector<int64_t> get_dims() const {
    return m_sizes;
  }

  bool empty() const {
    return (m_dim == 0);
  }
  void set_dim(size_t dim, int64_t size) {
    TORCH_CHECK(dim < m_dim, "dim idx is out of range");
    m_sizes[dim] = size;
  }
  void set_size(const std::vector<int64_t>& sizes);

  void set_scalar_type(at::ScalarType scalar_type) {
    is_scalar_initialized = true;
    scalar_type_ = scalar_type;
  }
  at::ScalarType get_scalar_type() {
    TORCH_CHECK(is_scalar_initialized, "Scalar Type is not initialized");
    return scalar_type_;
  }
  bool operator==(const TensorShape& shape) const {
    return (m_dim == shape.m_dim) && (n_elements == shape.n_elements) &&
        (m_sizes == shape.m_sizes);
  }
  bool operator!=(const TensorShape& shape) const {
    return !operator==(shape);
  }

  friend inline std::ostream& operator<<(
      std::ostream& O,
      const TensorShape& t) {
    O << '[';
    for (size_t i = 0; i < t.m_sizes.size(); i++) {
      O << (i > 0 ? ", " : "") << t.m_sizes[i];
    }
    O << ']';

    return O;
  }

  std::string DebugString() const {
    std::ostringstream sstr;
    sstr << "[";
    for (size_t i = 0; i < m_sizes.size(); i++) {
      sstr << (i > 0 ? ", " : "") << m_sizes[i];
    }
    sstr << "]";
    return sstr.str();
  }

  size_t Size() const {
    size_t size = sizeof(*this);
    size += m_sizes.size() * sizeof(decltype(m_sizes)::value_type);
    return size;
  }

  synTensorType get_tensor_type() const {
    return m_tensor_type;
  }

  void set_tensor_type(synTensorType tensor_type) {
    m_tensor_type = tensor_type;
  }

  void Serialize(std::ostream& os) const;
  TensorShape(std::istream& is);

 private:
  size_t m_dim{0};
  int64_t n_elements{0};
  bool is_scalar_initialized{false};
  at::ScalarType scalar_type_;
  std::vector<int64_t> m_sizes;
  synTensorType m_tensor_type{TENSOR_TYPE_INVALID};
};

} // namespace habana_helpers

CREATE_OSTREAM_FORMATTER(habana_helpers::TensorShape);
