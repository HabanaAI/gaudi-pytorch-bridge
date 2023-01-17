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

#include "tensor_shape.h"

#include <cassert>
#include <cstring>

#include <c10/util/Exception.h>

#include "habana_helpers/habana_serialization/include/habana_serialization/deserializers.h"
#include "habana_helpers/habana_serialization/include/habana_serialization/serializers.h"
namespace habana_helpers {

TensorShape::TensorShape(
    const at::IntArrayRef& sizes,
    at::ScalarType scalar_type) {
  m_sizes = sizes.vec();
  m_dim = m_sizes.size();
  n_elements = m_dim == 0 ? 0 : 1;
  for (size_t i = 0; i < m_dim; i++)
    n_elements *= m_sizes[i];
  scalar_type_ = scalar_type;
  is_scalar_initialized = true;
}

void TensorShape::add_dim(int64_t size) {
  m_sizes.emplace_back(size);
  m_dim++;
  n_elements = n_elements ? n_elements * size : size;
}

void TensorShape::set_size(const std::vector<int64_t>& sizes) {
  n_elements = sizes.size() == 0 ? 0 : 1;
  for (size_t i = 0; i < sizes.size(); i++)
    n_elements *= sizes[i];
  m_sizes = sizes;
  m_dim = sizes.size();
}

void TensorShape::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, m_dim);
  serialize(os, n_elements);
  serialize(os, is_scalar_initialized);
  serialize(os, scalar_type_);
  serialize(os, m_sizes);
  serialize(os, m_tensor_type);
}
TensorShape::TensorShape(std::istream& is) {
  using namespace serialization;
  deserialize(is, m_dim);
  deserialize(is, n_elements);
  deserialize(is, is_scalar_initialized);
  deserialize(is, scalar_type_);
  deserialize(is, m_sizes);
  deserialize(is, m_tensor_type);
}
} // namespace habana_helpers
