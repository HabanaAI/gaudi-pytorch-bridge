/******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <pytorch_helpers/habana_helpers/pt_version_check.h>
#include "generated/backend/_prelu_kernel.h"

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
      get_guid_with_precision("prelu_fwd", ScalarType()),
      {syn_in(0), reshape.get()},
      {{self.sizes(), self.scalar_type(), 0}});
  syn_out(0) = std::move(prelu[0]);
}

OutputMetaDataVector PreluBwdMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);
  const auto& weight = stack_tensor(stack, 2);

  OutputMetaDataVector meta(2);
  meta.at(0).shape = input.sizes().vec();
  meta.at(0).dtype = input.scalar_type();

  meta.at(1).shape = weight.sizes().vec();
  meta.at(1).dtype = weight.scalar_type();
  return meta;
}

} // namespace habana
