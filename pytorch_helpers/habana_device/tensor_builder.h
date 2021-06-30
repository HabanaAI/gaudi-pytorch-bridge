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

#include <absl/types/optional.h>
#include <c10/util/ArrayRef.h>
#include <synapse_helpers/habana_tensor.h>
#include <synapse_helpers/tensor_builder_base.h>
#include "habana_helpers/logging.h"
#include "synapse_helpers/type_conversions.h"

namespace synapse_helpers {
class tensor_builder : public tensor_builder_base<tensor_builder> {
 public:
  using tensor_builder_base::tensor_builder_base;

  // With strides
  explicit tensor_builder(
      const c10::IntArrayRef& shape,
      const c10::IntArrayRef& stride,
      synDataType data_type)
      : tensor_builder{shape.vec(), stride.vec(), data_type} {}

  explicit tensor_builder(
      const std::vector<int64_t>& shape,
      const std::vector<int64_t>& stride,
      synDataType data_type)
      : tensor_builder(
            to_shape_t(shape),
            to_stride_t(stride, shape, data_type),
            data_type) {}

  explicit tensor_builder(
      const tensor::shape_t& shape,
      const tensor::shape_t& stride,
      synDataType data_type)
      : tensor_builder_base(shape, stride, data_type) {}
};

}; // namespace synapse_helpers
