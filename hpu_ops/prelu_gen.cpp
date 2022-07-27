/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/hpu_op.h"

namespace habana {
void Prelu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  int64_t self_ndim = self.dim();
  std::vector<int64_t> reshape_shape(self_ndim, 1);
  // https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html#torch.nn.PReLU
  // Channel dim is the 2nd dim of input. When input has dims < 2, then there is
  // no channel dim and the number of channels = 1.
  if (self_ndim > 1) {
    reshape_shape[1] = self.size(1);
  }

  auto reshape =
      ReshapeHelper(graph, syn_in(1), reshape_shape, self.scalar_type());
  auto prelu = BuildOp(
      graph,
      "prelu_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), reshape.get()},
      {{self.sizes(), self.scalar_type(), 0}});
  syn_out(0) = std::move(prelu[0]);
}
} // namespace habana
