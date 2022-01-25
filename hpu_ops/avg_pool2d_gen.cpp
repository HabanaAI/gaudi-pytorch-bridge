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

#define CHECK_DIM(input_size)                                        \
  TORCH_CHECK(                                                       \
      input_size == 4,                                               \
      "Averagepool2D expects input_size equals to 4, but got size ", \
      input_size);

namespace habana {

std::shared_ptr<void> Fillavgpool2dParams(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> padding = {0, 0};
  auto kernel_size = stack.at(1).toIntVector();
  auto stride = stack.at(2).isNone() ? kernel_size : stack.at(2).toIntVector();
  auto pad = stack.at(3).isNone() ? padding : stack.at(3).toIntVector();
  const bool ceil_mode = stack.at(4).toBool();
  const bool include_pad = stack.at(5).toBool();

  PARAMS_STUB(ns_AveragePoolingWithDivisorOverride::Params);
  params->pad_w_begin = pad.size() == 1 ? pad.at(0) : pad.at(1);
  params->pad_w_end = pad.size() == 1 ? pad.at(0) : pad.at(1);
  params->pad_h_begin = pad.at(0);
  params->pad_h_end = pad.at(0);
  params->kernel_w =
      kernel_size.size() == 1 ? kernel_size.at(0) : kernel_size.at(1);
  params->kernel_h = kernel_size.at(0);
  params->stride_w = stride.size() == 1 ? stride.at(0) : stride.at(1);
  params->stride_h = stride.at(0);
  params->dilation_w = 1; // Dilation set to 1, since for AvgPool Pytorch API
                          // does not give dilation value
  params->dilation_h = 1;
  params->includePadding = include_pad ? 1 : 0;
  params->divisorOverride = stack.at(6).isNone() ? 0 : stack.at(6).toInt();
  params->pooling_convention = ceil_mode
      ? EPoolingConvention::POOLING_CONVENTION_FULL
      : EPoolingConvention::POOLING_CONVENTION_VALID;
  return params;
}

sizes_vec Avgpool2dOutputShape(const at::Stack& stack, bool) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  CHECK_DIM(self.dim());

  std::vector<long int> padding = {0, 0};
  auto kernel_size = stack.at(1).toIntVector();
  auto stride = stack.at(2).isNone() ? kernel_size : stack.at(2).toIntVector();
  auto pad = stack.at(3).isNone() ? padding : stack.at(3).toIntVector();

  const int filter_H = kernel_size.at(0);
  const int filter_W = kernel_size.size() == 1 ? filter_H : kernel_size.at(1);
  const int stride_H = stride.at(0);
  const int stride_W = stride.size() == 1 ? stride_H : stride.at(1);
  const int pad_H = pad.at(0);
  const int pad_W = pad.size() == 1 ? pad_H : pad.at(1);

  auto h_out = ((self.sizes()[2] + ((2 * pad_H) - filter_H)) / stride_H) + 1;
  auto w_out = ((self.sizes()[3] + ((2 * pad_W) - filter_W)) / stride_W) + 1;
  std::vector<int64_t> outshape{
      self.sizes()[0], // N
      self.sizes()[1], // C
      h_out,
      w_out,
  };
  return {outshape};
}

void Avgpool2d::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto input_shape = self.sizes();
  std::vector<int64_t> out_shape_4d = {
      input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
  size_t size = 0;
  const auto& params = Fillavgpool2dParams(stack, size);
  auto outshape = Avgpool2dOutputShape(stack)[0];

  synTransposeParams trans_params{};
  trans_params.tensorDim = self.dim();
  for (int i = 0; i < self.dim(); ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);

  auto transpose_nhwc = BuildOp(
      graph,
      "transpose",
      {syn_in(0)},
      {{out_shape_4d, ScalarType()}},
      &trans_params,
      sizeof(trans_params));

  std::vector<int64_t> avg_pool_size = {
      input_shape[0], outshape[2], outshape[3], input_shape[1]};
  auto resize = BuildOp(
      graph,
      "avg_pool_2d_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {transpose_nhwc[0].get()},
      {{avg_pool_size, ScalarType()}},
      params.get(),
      size);

  // Transpose N,H,W,C to N,C,H,W
  std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);

  auto transpose_nchw = BuildOp(
      graph,
      "transpose",
      {resize[0].get()},
      {{outshape, ScalarType(), 0}},
      &trans_params,
      sizeof(trans_params));
  syn_out(0) = std::move(transpose_nchw[0]);
}
} // namespace habana
