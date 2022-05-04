/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "reduction_template.h"

namespace habana {

void MinMaxOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();

  auto shape = MinMaxOutputShape(stack)[0];

  auto reduce_max = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {syn_in(0)},
      dim,
      keepdim,
      guid_,
      {{shape, ScalarType(), 0}, {shape, c10::ScalarType::Int, 1}});

  syn_out(0) = std::move(reduce_max[0]);
  syn_out(1) = std::move(reduce_max[1]);
}
} // namespace habana
