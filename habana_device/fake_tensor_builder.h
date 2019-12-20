#pragma once

#include <absl/types/optional.h>
#include <c10/util/ArrayRef.h>

#include <synapse_helpers/habana_tensor.h>

#define SYNAPSE_HELPERS_ASSERT_OPTIONAL(error_optional_for_eval)            \
  do {                                                                      \
    auto&& error_optional{error_optional_for_eval};                         \
    TORCH_CHECK(!error_optional.has_value(), error_optional.value().error); \
  } while (false)

namespace synapse_helpers {
class tensor_builder final {
  tensor_builder() = delete;

 public:
  static tensor create_tensor(
      synDeviceId device_id,
      synDataType data_type,
      uint64_t total_size_bytes,
      unsigned dimension_count,
      c10::IntArrayRef shape,
      std::string tensor_name,
      bool is_persistent) {
    return create_tensor(
        device_id,
        data_type,
        total_size_bytes,
        dimension_count,
        shape.vec(),
        tensor_name,
        is_persistent);
  }

  static tensor create_tensor(
      synDeviceId device_id,
      synDataType data_type,
      uint64_t total_size_bytes,
      unsigned dimension_count,
      std::vector<int64_t> shape,
      std::string tensor_name,
      bool is_persistent) {
    TORCH_CHECK(dimension_count == shape.size());
    TORCH_CHECK(
        shape.size() <= SYN_MAX_TENSOR_DIM,
        tensor_name,
        " tensor has more than ",
        SYN_MAX_TENSOR_DIM,
        " dimensions");

    tensor::dimension_sizes_t dimensions{};
    // write NHWC as CWHN and write 0 at the end
    for (int i = 0; i < shape.size(); ++i)
      dimensions[i] = shape[shape.size() - i - 1];

    tensor result_tensor(
        device_id,
        data_type,
        total_size_bytes,
        dimension_count,
        dimensions,
        tensor_name,
        is_persistent);

    auto error = result_tensor.create();
    SYNAPSE_HELPERS_ASSERT_OPTIONAL(error);

    return result_tensor;
  }
};

} // namespace synapse_helpers