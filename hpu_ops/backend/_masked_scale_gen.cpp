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

OutputMetaDataVector MaskedScaleMeta(const at::Stack& stack) {
  OutputMetaData meta;
  const torch::Tensor& self = stack_tensor(stack, 0);
  meta.dtype = self.scalar_type();
  meta.shape = self.sizes().vec();
  return {meta};
}

void MaskedScale::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto mask = stack.at(1).toTensor();
  auto scale = stack.at(2).toScalar().toDouble();
  scale = 1.0 / (1.0 - 1.0 / scale);
  const auto meta = MaskedScaleMeta(stack)[0];

  auto mult = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", meta.dtype),
      {syn_in(0), syn_in(1)},
      {{meta.shape, meta.dtype}});

  auto scale_tensor = ConstantHelper(graph, scale, meta.dtype, meta.shape);

  auto output = BuildOp(
      graph,
      get_guid_with_precision("mult_fwd", meta.dtype),
      {mult[0].get(), scale_tensor.get()},
      {{meta.shape, meta.dtype, 0}});

  syn_out(0) = std::move(output[0]);
}
} // namespace habana
