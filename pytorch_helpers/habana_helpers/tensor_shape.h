#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <vector>

namespace habana_helpers {

class TensorShape {
 public:
  TensorShape();
  TensorShape(const at::IntArrayRef& sizes, at::ScalarType scalar_type);
  void add_dim(int64_t size);
  int dims() const {
    return m_dim;
  }
  int64_t dim_size(int dim) const {
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
  void set_dim(int dim, int64_t size) {
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

 private:
  std::vector<int64_t> m_sizes;
  int m_dim;
  int64_t n_elements;
  at::ScalarType scalar_type_;
  bool is_scalar_initialized;
};

} // namespace habana_helpers
