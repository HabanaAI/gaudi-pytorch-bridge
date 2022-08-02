/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/reflection_pad1d.h"
#include "generated/reflection_pad1d_backward.h"
#include "hpu_op_helper.h"

namespace habana {

sizes_vec ReflectionPad1DOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> outputShape = self.sizes().vec();
  auto pad = stack.at(1).toIntVector();
  TORCH_CHECK((pad.size() == 2), "Pad size can only be 2 for ReflectionPad1d");
  // updating the width dimension
  outputShape.rbegin()[0] = outputShape.rbegin()[0] + pad[0] + pad[1];
  return {outputShape};
}

sizes_vec ReflectionPad2DOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> outputShape = self.sizes().vec();
  auto pad = stack.at(1).toIntVector();
  TORCH_CHECK((pad.size() == 4), "Pad size can only be 4 for ReflectionPad2d");
  // updating the width dimension
  outputShape.rbegin()[0] = outputShape.rbegin()[0] + pad[0] + pad[1];
  // updating the height dimension
  outputShape.rbegin()[1] = outputShape.rbegin()[1] + pad[2] + pad[3];
  return {outputShape};
}

sizes_vec ReflectionPad3DOutputShape(const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> outputShape = self.sizes().vec();
  auto pad = stack.at(1).toIntVector();
  TORCH_CHECK((pad.size() == 6), "Pad size can only be 6 for ReflectionPad3d");
  // updating the width dimension
  outputShape.rbegin()[0] = outputShape.rbegin()[0] + pad[0] + pad[1];
  // updating the height dimension
  outputShape.rbegin()[1] = outputShape.rbegin()[1] + pad[2] + pad[3];
  // updating the depth dimension
  outputShape.rbegin()[2] = outputShape.rbegin()[2] + pad[4] + pad[5];
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
  int mul = 0;
  int add = -1;
  // tpc kernel expects the pad before and pad after
  // for each dimension
  for (uint i = 0; i < pads.size(); i++) {
    if (i % 2 == 0) {
      mul = 0;
      add++;
    } else {
      mul = 1;
    }
    uint hpu_index = (mul * inputShape.size()) + add;
    params->pads[hpu_index] = pads[i];
  }
  return params;
}

std::shared_ptr<void> FillReflectionPadForwardParams(
    const at::Stack& stack,
    size_t& size) {
  // pad_index in the stack is 1 for forward ops
  return FillReflectionPadParams(stack, size, 1);
}

sizes_vec ReflectionPadBackwardOutputShape(const at::Stack& stack) {
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
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 1).sizes();
  size_t size = 0;
  const auto& params = FillReflectionPadBackwardParams(stack, size);
  // dropping off the second input to tpc kernel since it
  // expects only 1 input tensor
  auto reflection_pad = BuildOp(
      graph,
      "pad_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);

  // output
  syn_out(0) = std::move(reflection_pad[0]);
}

} // namespace habana
