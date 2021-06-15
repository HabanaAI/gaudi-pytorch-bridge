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

namespace synapse_helpers {
class tensor_builder : public tensor_builder_base<tensor_builder> {
 public:
  using tensor_builder_base::tensor_builder_base;

  explicit tensor_builder(const c10::IntArrayRef& shape, synDataType data_type)
      : tensor_builder{shape.vec(), data_type} {}

  explicit tensor_builder(
      const std::vector<int64_t>& shape,
      synDataType data_type)
      : tensor_builder(synapse_helpers::to_shape_t(shape), data_type) {}

  explicit tensor_builder(const tensor::shape_t& shape, synDataType data_type)
      : tensor_builder_base(shape, data_type) {}

  explicit tensor_builder(const tensor::shape_t& shape)
      : tensor_builder_base(shape) {}
};

}; // namespace synapse_helpers
