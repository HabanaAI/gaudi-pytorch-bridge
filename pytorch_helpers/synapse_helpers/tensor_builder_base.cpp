/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/tensor_builder_base.h"
#include "habana_helpers/logging.h"
#include "synapse_helpers/type_conversions.h"

#include <string>

namespace synapse_helpers {

tensor::shape_t to_shape_t(const std::vector<int64_t>& shape, bool reverse) {
  tensor::shape_t dimensions{tensor::shape_t::dimension_count_t{
      static_cast<unsigned>(shape.size())}}; // TODO make it more readable
  // write dimension backwards, e.g. NHWC as CWHN
  for (size_t i = 0; i < shape.size(); ++i) {
    dimensions[i] = reverse ? shape[shape.size() - i - 1] : shape[i];
  }

  return dimensions;
}

namespace detail {

std::string tensor_name_generator::generate() {
  return "tensor_" + std::to_string(id++);
}

void tensor_name_generator::reset() {
  id = 0;
}

uint64_t tensor_name_generator::id = 0;

uint64_t size_bytes_from_shape(
    const tensor::shape_t& shape,
    synDataType dataType) {
  HABANA_ASSERT(shape.rank().value <= 5U);
  uint64_t size = size_of_syn_data_type(dataType);
  for (auto i{0U}; i < shape.rank().value; ++i) {
    size *= shape[i];
  }
  return size;
}

} // namespace detail
} // namespace synapse_helpers
