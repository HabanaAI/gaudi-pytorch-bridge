/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <ATen/InferSize.h>
#include <ATen/native/Pool.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <algorithm>
#include <iostream>

#include "backend/create_pt_tensor.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_kernels/conv_pool_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/tensor_impl.h"

namespace habana {
/**
 * @brief Compute shape for output tensor(s) from given input tensor shape
 *         & pooling params such as kernel, stride, pad, dilation, ceil_mode
 */
std::vector<int64_t> PoolHelper::compute_output_shape(
    const at::Tensor& input,
    const at::IntArrayRef kernel_size,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode,
    bool is_input_nhwc = false) {
  const auto is_synapse_layout_handling_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING);

  const int filter_H = at::native::safe_downcast<int, int64_t>(kernel_size[0]);
  const int filter_W = kernel_size.size() == 1
      ? filter_H
      : at::native::safe_downcast<int, int64_t>(kernel_size[1]);

  const int stride_H = stride.empty()
      ? filter_H
      : at::native::safe_downcast<int, int64_t>(stride[0]);
  const int stride_W = stride.empty()
      ? filter_W
      : stride.size() == 1 ? stride_H
                           : at::native::safe_downcast<int, int64_t>(stride[1]);

  const int pad_H = at::native::safe_downcast<int, int64_t>(padding[0]);
  const int pad_W = padding.size() == 1
      ? pad_H
      : at::native::safe_downcast<int, int64_t>(padding[1]);

  const int dilation_H = at::native::safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_W = dilation.size() == 1
      ? dilation_H
      : at::native::safe_downcast<int, int64_t>(dilation[1]);

  // input NCHW, output NHWC
  // weight KCHW, where K - output channels
  // pad, stride HW
  unsigned int input_dim0 = 0;
  unsigned int input_dim1 = 1;
  unsigned int input_dim2 = 2;
  unsigned int input_dim3 = 3;

  if (is_synapse_layout_handling_enabled) {
    input_dim0 = synapse_helpers::layouts::INPUT_N_IDX;
    input_dim1 = synapse_helpers::layouts::INPUT_C_IDX;
    input_dim2 = synapse_helpers::layouts::INPUT_H_IDX;
    input_dim3 = synapse_helpers::layouts::INPUT_W_IDX;
  } else if (is_input_nhwc) {
    // If the input is already converted to NHWC, then the
    // input dimensions should be picked up in {0, 3, 1, 2}
    // order.
    input_dim0 = 0;
    input_dim1 = 3;
    input_dim2 = 1;
    input_dim3 = 2;
  }

  const int64_t N = input.size(input_dim0);
  const int64_t C = input.size(input_dim1);
  const int64_t input_H = input.size(input_dim2);
  const int64_t input_W = input.size(input_dim3);

  const int64_t output_H = at::native::pooling_output_shape<int64_t>(
      input_H, filter_H, pad_H, stride_H, dilation_H, ceil_mode);
  const int64_t output_W = at::native::pooling_output_shape<int64_t>(
      input_W, filter_W, pad_W, stride_W, dilation_W, ceil_mode);

  if (is_synapse_layout_handling_enabled) {
    return {N, C, output_H, output_W};
  } else {
    return {N, output_H, output_W, C};
  }
}

namespace {
/**
 * @brief Fill generic pooling params structure
 */
ns_SpatialReduction::Params synapse_pool_params_builder(
    const at::IntArrayRef& kernel_size, // HW
    const at::IntArrayRef& stride, // HW
    const at::IntArrayRef& padding, // HW
    const at::IntArrayRef& dilation, // HW
    bool ceil_mode) {
  const int64_t filter_H = kernel_size[0];
  const int64_t filter_W = kernel_size[1];
  // stride – the stride of the window. Default value is kernel_size
  const int64_t stride_H = stride.vec().empty() ? kernel_size[0] : stride[0];
  const int64_t stride_W = stride.vec().empty() ? kernel_size[1] : stride[1];
  const int64_t dilation_H = dilation[0];
  const int64_t dilation_W = dilation[1];

  ns_SpatialReduction::Params pool_params{};
  pool_params.kernel_w = filter_W;
  pool_params.kernel_h = filter_H;
  pool_params.stride_w = stride_W;
  pool_params.stride_h = stride_H;
  pool_params.pad_w_begin = padding[1];
  pool_params.pad_w_end = padding[1];
  pool_params.pad_h_begin = padding[0];
  pool_params.pad_h_end = padding[0];
  pool_params.dilation_w = dilation_W;
  pool_params.dilation_h = dilation_H;
  pool_params.pooling_convention =
      ceil_mode ? POOLING_CONVENTION_FULL : POOLING_CONVENTION_VALID;

  return pool_params;
}
} // namespace

OutputShapeInfRetType MaxPool2dWithIndicesOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  const auto dilation = inputs[4].toIntList().vec();
  bool ceil_mode = inputs[5].toBool();

  auto shape_out = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  OutputShapeInfRetType out;
  // output tensor
  out.AddOutputTensor(TensorMetaData(
      shape_out,
      HabanaOperator::CalculateStrides(
          shape_out, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));

  auto type = at::kByte;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    type = at::kShort;
  }
  // indices tensor
  out.AddOutputTensor(TensorMetaData(
      shape_out,
      HabanaOperator::CalculateStrides(
          shape_out, input.suggest_memory_format()),
      type,
      input.suggest_memory_format()));

  return out;
}

void MaxPool2dWithIndicesOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    at::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 6,
      "Incorrect size of input expected for MaxPool2dWithIndicesOperator");
  TORCH_CHECK(inputs[0].isTensor(), "First input type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isIntList(), "Second input type expected to be IntList");
  TORCH_CHECK(inputs[2].isIntList(), "Third input type expected to be IntList");
  TORCH_CHECK(
      inputs[3].isIntList(), "Fourth input type expected to be IntList");
  TORCH_CHECK(inputs[4].isIntList(), "Fifth input type expected to be IntList");
  TORCH_CHECK(inputs[5].isBool(), "Sixth input type expected to be Bool");
  TORCH_CHECK(
      output_metadata.size() == 2,
      "MaxPool2dWithIndicesOperator: #output_metadata should be 2");

  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  const auto dilation = inputs[4].toIntList().vec();
  bool ceil_mode = inputs[5].toBool();

  // Setup pool params
  auto syn_pool_params = synapse_pool_params_builder(
      kernel_size, stride, padding, dilation, ceil_mode);

  p_context_->params_.emplace<ns_SpatialReduction::Params>(syn_pool_params);
  p_context_->params_size_ = sizeof(syn_pool_params);

  auto out_shape = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  // Setup output tensors
  auto output_nhwc = habana::createPTTensor(
      input,
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options(),
      input.suggest_memory_format(),
      output_metadata.at(0).persistent);
  // NOTE: cpu and cuda implementations hold indices as kLong (int64). I am
  // using uint8 and short for float and bf16 input tensors respectively (to
  // match TPC kernel requirement).
  auto type = at::kByte;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    type = at::kShort;
  }

  auto output_idx_nhwc = habana::createPTTensor(
      input,
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options(),
      input.suggest_memory_format(),
      type,
      output_metadata.at(1).persistent);
  // The output_idx_nhwc is created with is_output_persistent[1] and
  // output_nhwc is created with is_output_persistent[0]. When adding
  // AllocateSynapseOutputs, the order is revered and the is_output_persistent
  // flag also need to be accordingly reversed.

  if (synapse_helpers::HPURegistrar::get_device().type() ==
      synDeviceType::synDeviceGreco) {
    AllocateSynapseOutput(graph, output_nhwc, output_metadata.at(0));
    AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
    p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
        output_idx_nhwc,
        graph,
        output_metadata.at(1).persistent,
        output_metadata.at(1).external,
        c10::nullopt,
        output_metadata.at(1).name));
    p_context_->pt_outputs_.emplace_back(output_idx_nhwc);
  } else {
    OutputMetaDataVector output_metadata_reordered = {
        output_metadata.at(1), output_metadata.at(0)};
    AllocateSynapseOutputs(
        graph, {output_idx_nhwc, output_nhwc}, output_metadata_reordered);
    AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
    std::swap(p_context_->pt_outputs_[0], p_context_->pt_outputs_[1]);
    std::swap(p_context_->syn_outputs_[0], p_context_->syn_outputs_[1]);
  }
}

void MaxPool2dWithIndicesOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  const auto dilation = inputs[4].toIntList().vec();
  bool ceil_mode = inputs[5].toBool();

  auto out_shape = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  // Setup output tensors
  auto output_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options());

  // NOTE: cpu and cuda implementations hold indices as kLong (int64). I am
  // using uint8 and short for float and bf16 input tensors respectively (to
  // match TPC kernel requirement).
  auto type = at::kByte;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    type = at::kShort;
  }
  auto output_idx_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options().dtype(type));
  std::vector<at::Tensor> v{output_nhwc, output_idx_nhwc};
  HabanaOperator::SetPTOutputs(v);
}
} // namespace habana

static auto& PoolKernelsKernelRegistry = habana::KernelRegistry().add(
    "aten::max_pool2d",
    KERNEL_FN(MaxPool2dOperator));
