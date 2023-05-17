/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/_masked_scale.h"

namespace habana {

sizes_vec MaskedScaleOutputShape(const at::Stack& stack) {
  const torch::Tensor& mask = stack_tensor(stack, 1);
  std::vector<int64_t> mask_shape = mask.sizes().vec();
  return {mask_shape};
}

void MaskedScale::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto mask = stack.at(1).toTensor();
  auto scale = stack.at(2).toScalar().toDouble();
  scale = 1.0 / (1.0 - 1.0 / scale);
  auto outshape = MaskedScaleOutputShape(stack)[0];

  auto mult = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", ScalarType()),
      {syn_in(0), syn_in(1)},
      {{outshape, ScalarType()}});

  auto scale_tensor = ConstantHelper(graph, scale, ScalarType(), outshape);

  auto output = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", ScalarType()),
      {mult[0].get(), scale_tensor.get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(output[0]);
}
} // namespace habana
