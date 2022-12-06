/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include "generated/_adaptive_avg_pool2d.h"
#include "generated/_adaptive_avg_pool2d_backward.h"
#include "generated/adaptive_avg_pool2d.h"

namespace habana {

std::shared_ptr<void> FillAdaptiveAvgPool2dParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  const auto output_size = stack[1].toIntList().vec();
  PARAMS_STUB(ns_AdaptiveAvgPool::Params);
  params->outputHeight = output_size[0];
  params->outputWidth = output_size[1];
  return params;
}

std::shared_ptr<void> FillAdaptiveAvgPool2dParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  const auto input = stack_tensor(stack, 1);
  PARAMS_STUB(ns_AdaptiveAvgPool::Params);
  params->outputHeight = input.size(-2);
  params->outputWidth = input.size(-1);
  return params;
}

sizes_vec AdaptiveAvgPool2dOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const auto output_size = stack[1].toIntList().vec();
  const auto input_size = self.dim();
  TORCH_CHECK(
      input_size == 4 || input_size == 3,
      "AdaptiveAvgPool2d expects input rank to be 4 or 3, but got size ",
      input_size);

  const int64_t output_H = output_size[0];
  const int64_t output_W = output_size.size() == 1 ? output_H : output_size[1];

  if (self.dim() == 4) {
    return {{self.size(0), self.size(1), output_H, output_W}};
  } else {
    return {{self.size(0), output_H, output_W}};
  }
}

sizes_vec AdaptiveAvgPool2dOutputShapeBwd(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  return {self.sizes().vec()};
}

void AdaptiveAvgPool2dBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillAdaptiveAvgPool2dParamsBwd(stack, size);
  auto outshape = AdaptiveAvgPool2dOutputShapeBwd(stack)[0];

  std::vector<synTensor> grad = {syn_in(0)};
  this->CreateShapeTensorInput(graph, this->ScalarType(), outshape, grad);
  auto adaptive_avg_pool = BuildOp(
      graph,
      "adaptive_avg_pool_2d_bwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      grad,
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(adaptive_avg_pool[0]);
}
} // namespace habana
