#pragma once

#include <absl/types/optional.h>
#include <c10/util/ArrayRef.h>
#include <synapse_helpers/habana_tensor.h>
#include <synapse_helpers/tensor_builder_base.h>

namespace synapse_helpers {
class tensor_builder : public tensor_builder_base<tensor_builder> {
  tensor::dimension_sizes_t to_dimension_sizes_t(
      const std::vector<int64_t>& shape) {
    TORCH_CHECK(
        shape.size() <= SYN_MAX_TENSOR_DIM,
        " tensor has more than ",
        SYN_MAX_TENSOR_DIM,
        " dimensions");

    tensor::dimension_sizes_t dimensions{};
    // write dimension backwards, e.g. NHWC as CWHN
    for (int i = 0; i < shape.size(); ++i)
      dimensions[i] = shape[shape.size() - i - 1];

    return dimensions;
  }

 public:
  using tensor_builder_base::tensor_builder_base;

  explicit tensor_builder(const c10::IntArrayRef& shape, synDataType data_type)
      : tensor_builder{shape.vec(), data_type} {}

  explicit tensor_builder(
      const std::vector<int64_t>& shape,
      synDataType data_type)
      : tensor_builder_base(
            to_dimension_sizes_t(shape),
            shape.size(),
            data_type) {}
};

}; // namespace synapse_helpers
