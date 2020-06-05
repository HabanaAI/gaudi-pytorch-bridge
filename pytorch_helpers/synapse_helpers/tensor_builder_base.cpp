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
#include "synapse_helpers/type_conversions.h"
#include "synapse_helpers/logging.h"

#include <string>

namespace synapse_helpers {
namespace detail {

std::string generate_name() {
  static uint64_t id = 0;
  return "tensor_" + std::to_string(id++);
}

uint64_t size_bytes_from_shape(const tensor::shape_t& shape, synDataType dataType) {
  HABANA_ASSERT(shape.rank().value <= 5U);
  uint64_t size = size_of_syn_data_type(dataType);
  for (auto i{0U}; i < shape.rank().value; ++i) {
    size *= shape[i];
  }
  return size;
}

}  // namespace detail
}  // namespace synapse_helpers
