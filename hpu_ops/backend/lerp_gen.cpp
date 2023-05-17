/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/lerp.h"

namespace habana {

sizes_vec LerpOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& end = stack_tensor(stack, 1);
  std::vector<std::vector<int64_t>> shape;
  if (stack.at(2).isTensor()) {
    const torch::Tensor& weight = stack_tensor(stack, 2);
    shape.emplace_back(at::infer_size(self.sizes(), weight.sizes()));
    shape.emplace_back(at::infer_size(shape[0], end.sizes()));
    return {shape[1]};
  }
  return {at::infer_size(self.sizes(), end.sizes())};
}

void Lerp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = LerpOutputShape(stack)[0];
  auto sub_outshape = at::infer_size(
      stack_tensor(stack, 0).sizes(), stack_tensor(stack, 1).sizes());

  // subtraction of start and end
  auto sub = BuildOp(
      graph,
      get_guid_with_precision("sub", ScalarType()),
      {syn_in(1), syn_in(0)},
      {{{sub_outshape}, ScalarType()}});

  // multiplication of weight and sub
  auto mult = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {syn_in(2), sub[0].get()},
      {{outshape, ScalarType()}});

  // addition of start and mult
  auto lerp = BuildOp(
      graph,
      get_guid_with_precision("add", ScalarType()),
      {syn_in(0), mult[0].get()},
      {{outshape, ScalarType(), 0}});

  // output of lerp is the output of this op
  syn_out(0) = std::move(lerp[0]);
}
} // namespace habana
