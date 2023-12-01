/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/reflection_pad1d.h"
#include "generated/backend/reflection_pad1d_backward.h"

namespace habana {

constexpr uint SELF_INDEX_FWD = 0;
constexpr uint PAD_INDEX_FWD = 1;
constexpr uint SELF_INDEX_BWD = 1;
constexpr uint PAD_INDEX_BWD = 2;
constexpr uint PADS_PER_DIM = 2;

sizes_vec ReflectionPadOutputShape(
    const at::Stack& stack,
    uint selfIndex,
    uint padIndex,
    uint dimsVariant) {
  std::vector<int64_t> outputShape =
      stack.at(selfIndex).toTensor().sizes().vec();
  auto pad = stack.at(padIndex).toIntVector();
  uint expectedPadsNumber = PADS_PER_DIM * dimsVariant;
  TORCH_CHECK(
      (pad.size() == expectedPadsNumber),
      "Pad size can only be %dd for ReflectionPad%dd",
      expectedPadsNumber,
      dimsVariant);

  for (uint dim = 0; dim < dimsVariant; dim++) {
    outputShape.rbegin()[dim] = outputShape.rbegin()[dim] +
        pad[dim * PADS_PER_DIM] + pad[dim * PADS_PER_DIM + 1];
  }
  return {outputShape};
}

OutputMetaDataVector ReflectionPadDMeta(
    const at::Stack& stack,
    uint dimsVariant) {
  auto self = stack.at(0).toTensor();
  OutputMetaData meta;

  meta.shape = ReflectionPadOutputShape(
      stack, SELF_INDEX_FWD, PAD_INDEX_FWD, dimsVariant)[0];
  meta.dtype = self.scalar_type();

  return {meta};
}

OutputMetaDataVector ReflectionPad1DMeta(const at::Stack& stack) {
  return ReflectionPadDMeta(stack, 1);
}

OutputMetaDataVector ReflectionPad2DMeta(const at::Stack& stack) {
  return ReflectionPadDMeta(stack, 2);
}

OutputMetaDataVector ReflectionPad3DMeta(const at::Stack& stack) {
  return ReflectionPadDMeta(stack, 3);
}

static std::shared_ptr<void> FillReflectionPadParams(
    const at::Stack& stack,
    size_t& size,
    uint self_index,
    uint pad_index) {
  PARAMS_STUB(ns_PadKernelEx::Params);
  auto self = stack.at(self_index).toTensor();
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
  return FillReflectionPadParams(stack, size, SELF_INDEX_FWD, PAD_INDEX_FWD);
}

std::shared_ptr<void> FillReflectionPadBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  return FillReflectionPadParams(stack, size, SELF_INDEX_BWD, PAD_INDEX_BWD);
}

OutputMetaDataVector ReflectionPadBackwardMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, SELF_INDEX_BWD);
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

void ReflectionPadBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillReflectionPadBackwardParams(stack, size);

  if (habana::ShapeInference::GetCurrentPass() ==
      habana::ShapeInfo::InferencePass::MAX_SHAPE) {
    synapse_helpers::tensor& syn_tensor_start = p_context_->syn_inputs_[0];
    std::vector<int64_t> max = std::get<1>(
        habana::ShapeInference::GetMinMaxShape(syn_tensor_start.id()));
    uint dimsVariant =
        stack.at(PAD_INDEX_BWD).toIntVector().size() / PADS_PER_DIM;
    auto outputShapeExpectedMax = ReflectionPadOutputShape(
        stack, SELF_INDEX_BWD, PAD_INDEX_BWD, dimsVariant);

    for (uint dim = 0; dim < max.size(); dim++) {
      TORCH_CHECK(
          (max[dim] <= outputShapeExpectedMax[0][dim]),
          "Output shape at dim=%d in max pass is greater than expectedd max output shape.",
          dim);
    }
  }
  auto meta = ReflectionPadBackwardMeta(stack)[0];

  // dropping off the second input to tpc kernel since it
  // expects only 1 input tensor
  auto reflection_pad = BuildOp(
      graph,
      get_guid_with_precision("pad_bwd", ScalarType()),
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  // output
  syn_out(0) = std::move(reflection_pad[0]);
}

} // namespace habana
