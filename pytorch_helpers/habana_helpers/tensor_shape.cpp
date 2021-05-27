#include "tensor_shape.h"
#include <c10/util/Exception.h>
#include <cassert>
#include <cstring>

namespace habana_helpers {

TensorShape::TensorShape()
    : m_sizes{}, m_dim(0), n_elements(0), is_scalar_initialized(false) {}

TensorShape::TensorShape(
    const at::IntArrayRef& sizes,
    at::ScalarType scalar_type) {
  m_sizes = sizes.vec();
  m_dim = m_sizes.size();
  n_elements = m_dim == 0 ? 0 : 1;
  for (auto i = 0; i < m_dim; i++)
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
  for (uint i = 0; i < sizes.size(); i++)
    n_elements *= sizes[i];
  m_sizes = sizes;
  m_dim = sizes.size();
}

} // namespace habana_helpers
