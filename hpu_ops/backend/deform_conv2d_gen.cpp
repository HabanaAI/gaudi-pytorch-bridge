/**
 * Copyright (c) 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <limits>
#include "generated/backend/_deform_conv2d_backward.h"
#include "generated/backend/deform_conv2d.h"

namespace habana {

OutputMetaDataVector DeformConv2dOutputMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto& weight = stack_tensor(stack, 1);
  const auto& offset = stack_tensor(stack, 2);
  const auto& mask = stack_tensor(stack, 3);
  const auto stride_h = stack[5].toInt();
  const auto stride_w = stack[6].toInt();
  const auto pad_h = stack[7].toInt();
  const auto pad_w = stack[8].toInt();
  const auto dilation_h = stack[9].toInt();
  const auto dilation_w = stack[10].toInt();
  const auto n_weight_grps = stack[11].toInt();
  const auto n_offset_grps = stack[12].toInt();
  const auto use_mask = stack[13].toBool();

  const int64_t batch_sz = input.size(0);
  const int64_t in_channels = input.size(1);
  const int64_t in_h = input.size(2);
  const int64_t in_w = input.size(3);

  const int64_t out_channels = weight.size(0);
  const int64_t weight_h = weight.size(2);
  const int64_t weight_w = weight.size(3);

  int64_t ker_h = dilation_h * (weight_h - 1) + 1;
  int64_t ker_w = dilation_w * (weight_w - 1) + 1;
  int64_t out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int64_t out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  OutputMetaData meta;
  meta.shape = {batch_sz, out_channels, out_h, out_w};
  meta.dtype = input.scalar_type();
  meta.mem_format = input.suggest_memory_format();

  HABANA_ASSERT(input.ndimension() == 4);
  HABANA_ASSERT(offset.ndimension() == 4);
  HABANA_ASSERT(!use_mask || mask.ndimension() == 4);
  HABANA_ASSERT(weight.ndimension() == 4);

  HABANA_ASSERT(
      weight_h > 0 && weight_w > 0,
      "weight_h: ",
      weight_h,
      " weight_w: ",
      weight_w);
  HABANA_ASSERT(
      stride_h == 1 && stride_w == 1,
      "Stride_h and stride_w must be equal to 1, got stride_h: ",
      stride_h,
      " stride_w: ",
      stride_w);
  HABANA_ASSERT(
      pad_h == 1 && pad_w == 1,
      "Pad_h and pad_w must be equal to 1, got pad_h: ",
      pad_h,
      " pad_w: ",
      pad_w);
  HABANA_ASSERT(
      dilation_h == 1 && dilation_w == 1,
      "Dilation_h and dilation_w must be equal to 1, got dilation_h: ",
      dilation_h,
      " dilation_w: ",
      dilation_w);
  HABANA_ASSERT(
      weight_h == 3 && weight_w == 3,
      "Weight_h and weight_w must be equal to 3, got weight_h: ",
      weight_h,
      " weight_w: ",
      weight_w);

  HABANA_ASSERT(weight.size(1) * n_weight_grps == input.size(1));
  HABANA_ASSERT(weight.size(0) % n_weight_grps == 0);
  HABANA_ASSERT(
      (offset.size(1) == n_offset_grps * 2 * weight_h * weight_w),
      "offset.shape[1] is not valid: got: ",
      offset.size(1),
      " expected: ",
      n_offset_grps * 2 * weight_h * weight_w);
  HABANA_ASSERT(
      (!use_mask || mask.size(1) == n_offset_grps * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ",
      mask.size(1),
      " expected: ",
      n_offset_grps * weight_h * weight_w);
  HABANA_ASSERT(input.size(1) % n_offset_grps == 0);

  HABANA_ASSERT(
      (offset.size(0) == input.size(0)), "invalid batch size of offset");
  HABANA_ASSERT(
      (offset.size(2) == out_h && offset.size(3) == out_w),
      "offset output dims: (",
      offset.size(2),
      ", ",
      offset.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  HABANA_ASSERT((mask.size(0) == input.size(0)), "invalid batch size of mask");
  HABANA_ASSERT(
      (!use_mask || (mask.size(2) == out_h && mask.size(3) == out_w)),
      "mask output dims: (",
      mask.size(2),
      ", ",
      mask.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  HABANA_ASSERT(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ",
      out_h,
      " out_w: ",
      out_w);
  HABANA_ASSERT(
      in_channels == out_channels,
      "In channels and out channels must be equal. Got in_channels: ",
      in_channels,
      ", out_channels: ",
      out_channels);

  return {meta};
}

using namespace std::literals;

void DeformConv2d::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);

  std::vector<synTensor> syn_inputs{
      syn_in(0), syn_in(2), syn_in(1), syn_in(3), syn_in(4)};
  ns_DeformConv::Params params{};

  const auto stride_w = stack[6].toInt();
  const auto stride_h = stack[5].toInt();
  const auto pad_w = stack[8].toInt();
  const auto pad_h = stack[7].toInt();
  const auto dilation_w = stack[10].toInt();
  const auto dilation_h = stack[9].toInt();

  HABANA_ASSERT(
      stride_w >= std::numeric_limits<int>::min() &&
          stride_w <= std::numeric_limits<int>::max(),
      "stride_w out of range for int conversion: ",
      stride_w);
  HABANA_ASSERT(
      stride_h >= std::numeric_limits<int>::min() &&
          stride_h <= std::numeric_limits<int>::max(),
      "stride_h out of range for int conversion: ",
      stride_h);
  HABANA_ASSERT(
      pad_w >= std::numeric_limits<int>::min() &&
          pad_w <= std::numeric_limits<int>::max(),
      "pad_w out of range for int conversion: ",
      pad_w);
  HABANA_ASSERT(
      pad_h >= std::numeric_limits<int>::min() &&
          pad_h <= std::numeric_limits<int>::max(),
      "pad_h out of range for int conversion: ",
      pad_h);
  HABANA_ASSERT(
      dilation_w >= std::numeric_limits<int>::min() &&
          dilation_w <= std::numeric_limits<int>::max(),
      "dilation_w out of range for int conversion: ",
      dilation_w);
  HABANA_ASSERT(
      dilation_h >= std::numeric_limits<int>::min() &&
          dilation_h <= std::numeric_limits<int>::max(),
      "dilation_h out of range for int conversion: ",
      dilation_h);

  params.strideW = static_cast<int>(stride_w);
  params.strideH = static_cast<int>(stride_h);
  params.padW = static_cast<int>(pad_w);
  params.padH = static_cast<int>(pad_h);
  params.dilationW = static_cast<int>(dilation_w);
  params.dilationH = static_cast<int>(dilation_h);

  auto meta = DeformConv2dOutputMeta(stack)[0];
  syn_out(0) = std::move(BuildOp(
      graph,
      get_guid_with_precision("deform_conv"sv, input.scalar_type()),
      std::move(syn_inputs),
      {{meta.shape, meta.dtype, 0}},
      &params,
      sizeof(params))[0]);
}

OutputMetaDataVector DeformConv2dBackwardOutputMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);
  const auto& weight = stack_tensor(stack, 2);
  const auto& offset = stack_tensor(stack, 3);
  const auto& mask = stack_tensor(stack, 4);
  const auto& bias = stack_tensor(stack, 5);

  OutputMetaData meta_input{input.scalar_type(), input.sizes().vec()};
  OutputMetaData meta_weight{weight.scalar_type(), weight.sizes().vec()};
  OutputMetaData meta_offset{offset.scalar_type(), offset.sizes().vec()};
  OutputMetaData meta_mask{mask.scalar_type(), mask.sizes().vec()};
  OutputMetaData meta_bias{bias.scalar_type(), bias.sizes().vec()};

  return {meta_input, meta_weight, meta_offset, meta_mask, meta_bias};
}

void DeformConv2dBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);

  std::vector<synTensor> syn_inputs{
      syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4)};

  auto meta = DeformConv2dBackwardOutputMeta(stack);
  auto grads = BuildOp(
      graph,
      get_guid_with_precision("deform_conv_bwd"sv, input.scalar_type()),
      std::move(syn_inputs),
      {{meta[0].shape, meta[0].dtype, 0},
       {meta[1].shape, meta[1].dtype, 1},
       {meta[2].shape, meta[2].dtype, 2},
       {meta[3].shape, meta[3].dtype, 3},
       {meta[4].shape, meta[4].dtype, 4}});

  syn_out(0) = std::move(grads[0]);
  syn_out(1) = std::move(grads[1]);
  syn_out(2) = std::move(grads[2]);
  syn_out(3) = std::move(grads[3]);
  syn_out(4) = std::move(grads[4]);
}

} // namespace habana
