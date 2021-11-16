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
#include "hpu_op_helper.h"

namespace habana {

sizes_vec ReflectionPad1DOutputShape(const at::Stack& stack, bool) {
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> outputShape = self.sizes().vec();
  auto pad = stack.at(1).toIntVector();
  // updating the width dimension
  outputShape.rbegin()[0] =
      outputShape.rbegin()[0] + pad[0] + pad[outputShape.size()];
  return {outputShape};
}

sizes_vec ReflectionPad2DOutputShape(const at::Stack& stack, bool) {
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> outputShape = self.sizes().vec();
  auto pad = stack.at(1).toIntVector();
  // updating the width dimension
  outputShape.rbegin()[0] =
      outputShape.rbegin()[0] + pad[0] + pad[outputShape.size()];
  // updating the height dimension
  outputShape.rbegin()[1] =
      outputShape.rbegin()[1] + pad[1] + pad[outputShape.size() + 1];
  return {outputShape};
}

static std::shared_ptr<void> FillReflectionPadParams(
    const at::Stack& stack,
    size_t& size,
    uint pad_index) {
  PARAMS_STUB(ns_PadKernelEx::Params);
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> inputShape = self.sizes().vec();
  auto pads = stack.at(pad_index).toIntVector();
  params->mode = PadMode_t::PAD_MODE_REFLECT;
  TORCH_CHECK(
      (pads.size() >= (2 * inputShape.size())),
      "Pads size (",
      pads.size(),
      ") is less than 2 * input's size (",
      2 * inputShape.size(),
      ")");
  for (uint i = 0; i < (2 * inputShape.size()); i++)
    params->pads[i] = pads[i];
  return params;
}

std::shared_ptr<void> FillReflectionPadForwardParams(
    const at::Stack& stack,
    size_t& size) {
  // pad_index in the stack is 1 for forward ops
  return FillReflectionPadParams(stack, size, 1);
}

sizes_vec ReflectionPadBackwardOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 1);
  return {self.sizes().vec()};
}

std::shared_ptr<void> FillReflectionPadBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  // pad_index in the stack is 2 for backward ops
  return FillReflectionPadParams(stack, size, 2);
}

void ReflectionPad::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const auto& outshape = stack_tensor(stack, 1).sizes();
  size_t size = 0;
  const auto& params = FillReflectionPadBackwardParams(stack, size);
  // dropping off the second input to tpc kernel since it
  // expects only 1 input tensor
  auto reflection_pad = BuildOp(
      graph,
      "pad_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}},
      params.get(),
      size);

  // output
  syn_out(0) = std::move(reflection_pad[0]);
}

} // namespace habana