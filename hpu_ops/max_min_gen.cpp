/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/max.h"
#include "reduction_template.h"

namespace habana {

sizes_vec MinMaxOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();

  auto shapes = ReductionOutputShape(self, dim, keepdim)[0];
  return {shapes, shapes};
}

std::shared_ptr<void> FillMinMaxParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_Reduction::Params);
  auto dim = stack.at(1).toInt();
  dim = (dim >= 0) ? static_cast<int>(stack.at(0).toTensor().dim()) - 1 - dim
                   : -(dim + 1);

  params->reductionDimension = dim;
  return params;
}

void MinMaxOut::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto keepdim = stack.at(2).toBool();

  auto shape = MinMaxOutputShape(stack, true)[0];

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
