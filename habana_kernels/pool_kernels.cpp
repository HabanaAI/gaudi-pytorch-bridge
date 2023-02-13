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
// #include <ATen/native/Pool.h> // TODO: fix this include
#include <ATen/div_rtn.h> // TODO: remove this header after ATen/native/Pool.h is included
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
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/tensor_impl.h"

using namespace torch;
using namespace habana;

namespace { // Copy paste from ATen/native/Pool.h
template <typename dest_t, typename src_t>
static inline dest_t safe_downcast(src_t v) {
  TORCH_CHECK(
      std::numeric_limits<dest_t>::min() <= v &&
          v <= std::numeric_limits<dest_t>::max(),
      "integer out of range");

  return static_cast<dest_t>(v);
}

template <typename T>
static inline T pooling_output_shape_pad_lr(
    T inputSize,
    T kernelSize,
    T pad_l,
    T pad_r,
    T stride,
    T dilation,
    bool ceil_mode) {
  T outputSize = div_rtn<T>(
                     inputSize + pad_l + pad_r - dilation * (kernelSize - 1) -
                         1 + (ceil_mode ? stride - 1 : 0),
                     stride) +
      1;
  if (pad_l) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputSize - 1) * stride >= inputSize + pad_l)
      --outputSize;
  }
  return outputSize;
}

template <typename T>
static inline T pooling_output_shape(
    T inputSize,
    T kernelSize,
    T pad,
    T stride,
    T dilation,
    bool ceil_mode) {
  return pooling_output_shape_pad_lr(
      inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}
} // namespace

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
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    return PoolHelper::compute_output_shape_synapse(
        input, kernel_size, stride, padding, dilation, ceil_mode);
  }
  const int filter_H = safe_downcast<int, int64_t>(kernel_size[0]);
  const int filter_W = kernel_size.size() == 1
      ? filter_H
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int stride_H =
      stride.empty() ? filter_H : safe_downcast<int, int64_t>(stride[0]);
  const int stride_W = stride.empty()
      ? filter_W
      : stride.size() == 1 ? stride_H : safe_downcast<int, int64_t>(stride[1]);

  const int pad_H = safe_downcast<int, int64_t>(padding[0]);
  const int pad_W =
      padding.size() == 1 ? pad_H : safe_downcast<int, int64_t>(padding[1]);

  const int dilation_H = safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_W = dilation.size() == 1
      ? dilation_H
      : safe_downcast<int, int64_t>(dilation[1]);

  // input NCHW, output NHWC
  // weight KCHW, where K - output channels
  // pad, stride HW
  unsigned int input_dim0 = 0;
  unsigned int input_dim1 = 1;
  unsigned int input_dim2 = 2;
  unsigned int input_dim3 = 3;

  // If the input is already converted to NHWC, then the
  // input dimensions should be picked up in {0, 3, 1, 2}
  // order.
  if (is_input_nhwc) {
    input_dim0 = 0;
    input_dim1 = 3;
    input_dim2 = 1;
    input_dim3 = 2;
  }
  const int64_t N = input.size(input_dim0);
  const int64_t C = input.size(input_dim1);
  const int64_t input_H = input.size(input_dim2);
  const int64_t input_W = input.size(input_dim3);
  const int64_t output_H = pooling_output_shape<int64_t>(
      input_H, filter_H, pad_H, stride_H, dilation_H, ceil_mode);
  const int64_t output_W = pooling_output_shape<int64_t>(
      input_W, filter_W, pad_W, stride_W, dilation_W, ceil_mode);

  std::vector<int64_t> outshape{N, output_H, output_W, C};
  return outshape;
}

/**
 * @brief Compute shape for output tensor(s) from given input tensor shape
 *         & pooling params such as kernel, stride, pad, dilation, ceil_mode
 */
std::vector<int64_t> PoolHelper::compute_output_shape(
    const at::Tensor& input,
    const at::IntArrayRef output_size,
    bool is_input_nhwc = false) {
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    return PoolHelper::compute_output_shape_synapse(input, output_size);
  }
  const int output_H = safe_downcast<int, int64_t>(output_size[0]);
  const int output_W = output_size.size() == 1
      ? output_H
      : safe_downcast<int, int64_t>(output_size[1]);

  // If the input is already converted to NHWC, then the
  // input dimensions should be picked up in {0, 3, 1, 2}
  // order.
  unsigned int input_dim0 = 0;
  unsigned int input_dim1 = is_input_nhwc ? 3 : 1;

  const int64_t N = input.size(input_dim0);
  const int64_t C = input.size(input_dim1);

  std::vector<int64_t> outshape{N, output_H, output_W, C};
  return outshape;
}

std::vector<int64_t> PoolHelper::compute_output_shape_synapse(
    const at::Tensor& input,
    const at::IntArrayRef kernel_size,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape for Synapse layout handling mode");

  const int filter_H = safe_downcast<int, int64_t>(kernel_size[0]);
  const int filter_W = kernel_size.size() == 1
      ? filter_H
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int stride_H =
      stride.empty() ? filter_H : safe_downcast<int, int64_t>(stride[0]);
  const int stride_W = stride.empty()
      ? filter_W
      : stride.size() == 1 ? stride_H : safe_downcast<int, int64_t>(stride[1]);

  const int pad_H = safe_downcast<int, int64_t>(padding[0]);
  const int pad_W =
      padding.size() == 1 ? pad_H : safe_downcast<int, int64_t>(padding[1]);

  const int dilation_H = safe_downcast<int, int64_t>(dilation[0]);
  const int dilation_W = dilation.size() == 1
      ? dilation_H
      : safe_downcast<int, int64_t>(dilation[1]);

  const int64_t N = input.size(synapse_helpers::layouts::INPUT_N_IDX);
  const int64_t C = input.size(synapse_helpers::layouts::INPUT_C_IDX);
  const int64_t input_H = input.size(synapse_helpers::layouts::INPUT_H_IDX);
  const int64_t input_W = input.size(synapse_helpers::layouts::INPUT_W_IDX);
  const int64_t output_H = pooling_output_shape<int64_t>(
      input_H, filter_H, pad_H, stride_H, dilation_H, ceil_mode);
  const int64_t output_W = pooling_output_shape<int64_t>(
      input_W, filter_W, pad_W, stride_W, dilation_W, ceil_mode);

  std::vector<int64_t> outshape{N, C, output_H, output_W};
  return outshape;
}

std::vector<int64_t> PoolHelper::compute_output_shape_synapse(
    const at::Tensor& input,
    const at::IntArrayRef output_size) {
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape for Synapse layout handling mode");

  const int output_H = safe_downcast<int, int64_t>(output_size[0]);
  const int output_W = output_size.size() == 1
      ? output_H
      : safe_downcast<int, int64_t>(output_size[1]);

  const int64_t N = input.size(synapse_helpers::layouts::INPUT_N_IDX);
  const int64_t C = input.size(synapse_helpers::layouts::INPUT_C_IDX);

  std::vector<int64_t> outshape{N, C, output_H, output_W};
  return outshape;
}

/**
 * @brief Fill generic pooling params structure
 */
ns_SpatialReduction::Params synapse_pool_params_builder(
    const IntArrayRef& kernel_size, // HW
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation, // HW
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

  auto type = kByte;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    type = kShort;
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

/**
 * @brief Fill Average pooling params structure
 */
ns_AveragePooling::Params synapse_avg_pool_params_builder(
    const IntArrayRef& kernel_size, // HW
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation, // HW
    int include_padding) {
  ns_SpatialReduction::Params* pt_pool_params;
  ns_AveragePooling::Params avg_pool_params{};
  pt_pool_params = &avg_pool_params;
  *pt_pool_params = synapse_pool_params_builder(
      kernel_size, stride, padding, dilation, false);
  avg_pool_params.includePadding = include_padding;

  return avg_pool_params;
}

/**
 * @brief Fill Adaptive Average pooling params structure
 */
ns_AdaptiveAvgPool::Params synapse_adaptive_avg_pool_params_builder(
    const IntArrayRef& output_size) {
  ns_AdaptiveAvgPool::Params adaptive_avg_pool_params{};
  adaptive_avg_pool_params.outputHeight = output_size[0];
  adaptive_avg_pool_params.outputWidth = output_size[1];

  return adaptive_avg_pool_params;
}

void MaxPool2dWithIndicesOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
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
  auto type = kByte;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    type = kShort;
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

habana::OutputShapeInfRetType MaxPool2dWithIndicesBackwardOutOperator::
    ComputeOutputShape(torch::jit::Stack& inputs) {
  at::Tensor grad_input = inputs[0].toTensor();

  OutputShapeInfRetType out;
  auto tensor_meta_data = TensorMetaData(
      grad_input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          grad_input.sizes().vec(), grad_input.suggest_memory_format()),
      grad_input.scalar_type(),
      grad_input.suggest_memory_format());
  // output tensor
  out.AddOutputTensor(tensor_meta_data);
  out.AddShapeTensor(tensor_meta_data);
  return out;
}

void MaxPool2dWithIndicesBackwardOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of input expected for MaxPool2dWithIndicesBackwardOutOperator");
  TORCH_CHECK(inputs[0].isTensor(), "First input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Second input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Third input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Fourth input type expected to be tensor");
  TORCH_CHECK(inputs[4].isIntList(), "Fifth input type expected to be IntList");
  TORCH_CHECK(inputs[5].isIntList(), "Sixth input type expected to be IntList");
  TORCH_CHECK(
      inputs[6].isIntList(), "Seventh input type expected to be IntList");
  TORCH_CHECK(
      inputs[7].isIntList(), "Eighth input type expected to be IntList");
  TORCH_CHECK(inputs[8].isBool(), "Ninth input type expected to be Bool");

  at::Tensor grad_input = inputs[0].toTensor();
  at::Tensor grad_out = inputs[1].toTensor();
  at::Tensor input = inputs[2].toTensor();
  at::Tensor indices = inputs[3].toTensor();
  const auto kernel_size = inputs[4].toIntList().vec();
  const auto stride = inputs[5].toIntList().vec();
  const auto padding = inputs[6].toIntList().vec();
  const auto dilation = inputs[7].toIntList().vec();
  bool ceil_mode = inputs[8].toBool();

  // Setup pool params
  auto syn_pool_params = synapse_pool_params_builder(
      kernel_size, stride, padding, dilation, ceil_mode);

  p_context_->params_.emplace<ns_SpatialReduction::Params>(syn_pool_params);
  p_context_->params_size_ = sizeof(syn_pool_params);

  auto out_shape = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[1], out_shape[2], out_shape[3]};
  TORCH_CHECK(
      grad_out.sizes().vec() == expected_output_size,
      " expected:",
      grad_out.sizes().vec(),
      " but got: ",
      expected_output_size);
  TORCH_CHECK(input.sizes() == grad_input.sizes());
  TORCH_CHECK(grad_out.sizes() == indices.sizes());

  TORCH_CHECK(
      (indices.scalar_type() == c10::ScalarType::Byte) ||
      (indices.scalar_type() == c10::ScalarType::Short));

  // Allocate Shape tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, grad_input);
  }

  AllocateSynapseOutput(graph, {grad_input}, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
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
  auto type = kByte;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    type = kShort;
  }
  auto output_idx_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options().dtype(type));
  std::vector<at::Tensor> v{output_nhwc, output_idx_nhwc};
  HabanaOperator::SetPTOutputs(v);
}

void MaxPool2dWithIndicesBackwardOutOperator::SetPTOutputs(
    torch::jit::Stack& inputs) {
  at::Tensor grad_input = inputs[0].toTensor();
  at::Tensor grad_out = inputs[1].toTensor();
  at::Tensor input = inputs[2].toTensor();
  at::Tensor indices = inputs[3].toTensor();
  const auto kernel_size = inputs[4].toIntList().vec();
  const auto stride = inputs[5].toIntList().vec();
  const auto padding = inputs[6].toIntList().vec();
  const auto dilation = inputs[7].toIntList().vec();
  bool ceil_mode = inputs[8].toBool();

  auto out_shape = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[1], out_shape[2], out_shape[3]};
  TORCH_CHECK(
      grad_out.sizes().vec() == expected_output_size,
      " expected: ",
      grad_out.sizes().vec(),
      " but got: ",
      expected_output_size);
  TORCH_CHECK(input.sizes() == grad_input.sizes());
  TORCH_CHECK(grad_out.sizes() == indices.sizes());

  TORCH_CHECK(
      (indices.scalar_type() == c10::ScalarType::Byte) ||
      (indices.scalar_type() == c10::ScalarType::Short));
  std::vector<at::Tensor> v{grad_input};
  HabanaOperator::SetPTOutputs(v);
}

habana::OutputShapeInfRetType MaxPool2dWithIndicesBackwardOperator::
    ComputeOutputShape(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[1].toTensor();
  auto grad_input = habana::createPTTensor(input, false);

  // Re-order the inputs for:
  // MaxPool2dWithIndicesBackwardOutOperator in the below order:
  // {grad_input, grad_out, input, indices, kernel_size, stride, padding,
  //  dialation, ceil_mode}
  inputs.insert(inputs.begin(), IValue(grad_input));
  auto& indices = inputs.back();
  inputs.insert(inputs.begin() + 3, indices);
  inputs.pop_back();
  return MaxPool2dWithIndicesBackwardOutOperator::ComputeOutputShape(inputs);
}

void MaxPool2dWithIndicesBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of input expected for MaxPool2dWithIndicesBackwardOperator");
  TORCH_CHECK(inputs[1].isTensor(), "Second input type expected to be tensor");

  at::Tensor input = inputs[1].toTensor();
  auto grad_input =
      habana::createPTTensor(input, output_metadata.at(0).persistent);

  // Re-order the inpust for:
  // MaxPool2dWithIndicesBackwardOutOperator in the below order:
  // {grad_input, grad_out, input, indices, kernel_size, stride, padding,
  //  dialation, ceil_mode}
  inputs.insert(inputs.begin(), IValue(grad_input));
  auto& indices = inputs.back();
  inputs.insert(inputs.begin() + 3, indices);
  inputs.pop_back();

  MaxPool2dWithIndicesBackwardOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, output_metadata);

  // Revert to the original input stack
  inputs.push_back(indices);
  inputs.erase(inputs.begin() + 3);
  inputs.erase(inputs.begin());
}

void MaxPool2dWithIndicesBackwardOperator::SetPTOutputs(
    torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of input expected for MaxPool2dWithIndicesBackwardOperator");
  TORCH_CHECK(inputs[1].isTensor(), "Second input type expected to be tensor");

  at::Tensor input = inputs[1].toTensor();
  auto grad_input =
      at::empty_like(input, input.options(), input.suggest_memory_format());
  inputs.insert(inputs.begin(), IValue(grad_input));
  auto& indices = inputs.back();
  inputs.insert(inputs.begin() + 3, indices);
  inputs.pop_back();

  MaxPool2dWithIndicesBackwardOutOperator::SetPTOutputs(inputs);
}

void AvgPool2dOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input0 type expected to be tensor");
  TORCH_CHECK(inputs[1].isIntList(), "Input1 type expected to be IntList");
  TORCH_CHECK(inputs[2].isIntList(), "Input2 type expected to be IntList");
  TORCH_CHECK(inputs[3].isIntList(), "Input3 type expected to be IntList");
  TORCH_CHECK(inputs[4].isBool(), "Input4 type expected to be Bool");
  TORCH_CHECK(inputs[5].isBool(), "Input5 type expected to be Bool");

  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  auto ceil_mode = inputs[4].toBool();
  auto count_include_pad = inputs[5].toBool();
  auto divisor_override = inputs[6].toOptional<int64_t>();

  TORCH_CHECK(
      !divisor_override.has_value(),
      "avgpool_2d: divisor override is not supported");

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  // Setup pool params
  auto syn_pool_params = synapse_avg_pool_params_builder(
      kernel_size, stride, padding, dilation, count_include_pad);

  p_context_->params_.emplace<ns_AveragePooling::Params>(syn_pool_params);
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

  AllocateSynapseOutput(graph, output_nhwc, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
}

void AvgPool2dOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  auto ceil_mode = inputs[4].toBool();
  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  auto out_shape = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  // Setup output tensors
  auto output_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options());
  std::vector<at::Tensor> v{output_nhwc};
  HabanaOperator::SetPTOutputs(v);
}

void AvgPool2dBackwardOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[0].isTensor(), "Input0 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input1 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input2 type expected to be tensor");
  TORCH_CHECK(inputs[3].isIntList(), "Input3 type expected to be IntList");
  TORCH_CHECK(inputs[4].isIntList(), "Input4 type expected to be IntList");
  TORCH_CHECK(inputs[5].isIntList(), "Input5 type expected to be IntList");
  TORCH_CHECK(inputs[6].isBool(), "Input6 type expected to be Bool");
  TORCH_CHECK(inputs[7].isBool(), "Input7 type expected to be Bool");

  at::Tensor grad_input_nhwc = inputs[0].toTensor();
  at::Tensor grad_out_nhwc = inputs[1].toTensor();
  at::Tensor input_nhwc = inputs[2].toTensor();
  const auto kernel_size = inputs[3].toIntList().vec();
  const auto stride = inputs[4].toIntList().vec();
  const auto padding = inputs[5].toIntList().vec();
  auto ceil_mode = inputs[6].toBool();
  auto count_include_pad = inputs[7].toBool();
  auto divisor_override = inputs[8].toOptional<int64_t>();

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  auto out_shape = PoolHelper::compute_output_shape(
      input_nhwc, kernel_size, stride, padding, dilation, ceil_mode, true);
  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[1], out_shape[2], out_shape[3]};

  TORCH_CHECK(
      !divisor_override.has_value(),
      "avgpool_2d: divisor override is not supported");
  TORCH_CHECK(
      input_nhwc.sizes() == grad_input_nhwc.sizes(),
      "Input and grad_input sizes don't match");
  TORCH_CHECK(grad_out_nhwc.sizes().vec() == expected_output_size);

  // Setup pool params
  auto syn_pool_params = synapse_avg_pool_params_builder(
      kernel_size, stride, padding, dilation, count_include_pad);

  p_context_->params_.emplace<ns_AveragePooling::Params>(syn_pool_params);
  p_context_->params_size_ = sizeof(syn_pool_params);

  // Allocate Shape tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, grad_input_nhwc);
  }

  AllocateSynapseOutput(graph, grad_input_nhwc, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
}

void AvgPool2dBackwardOutOperator::SetPTOutputs(Stack& inputs) {
  at::Tensor grad_input_nhwc = inputs[0].toTensor();
  at::Tensor grad_out_nhwc = inputs[1].toTensor();
  at::Tensor input_nhwc = inputs[2].toTensor();
  const auto kernel_size = inputs[3].toIntList().vec();
  const auto stride = inputs[4].toIntList().vec();
  const auto padding = inputs[5].toIntList().vec();
  auto ceil_mode = inputs[6].toBool();

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  auto out_shape = PoolHelper::compute_output_shape(
      input_nhwc, kernel_size, stride, padding, dilation, ceil_mode, true);
  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[1], out_shape[2], out_shape[3]};

  TORCH_CHECK(
      input_nhwc.sizes() == grad_input_nhwc.sizes(),
      "Input and grad_input sizes don't match");
  TORCH_CHECK(grad_out_nhwc.sizes().vec() == expected_output_size);
  std::vector<at::Tensor> v{grad_input_nhwc};
  HabanaOperator::SetPTOutputs(v);
}

void AvgPool2dBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(inputs[1].isTensor(), "Input1 type expected to be tensor");

  at::Tensor input_nhwc = inputs[1].toTensor();
  auto grad_input_nhwc =
      habana::createPTTensor(input_nhwc, output_metadata.at(0).persistent);

  inputs.insert(inputs.begin(), IValue(grad_input_nhwc));
  AvgPool2dBackwardOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, output_metadata);
}

void AvgPool2dBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input_nhwc = inputs[1].toTensor();
  auto grad_input_nhwc = at::empty_like(input_nhwc, input_nhwc.options());

  inputs.insert(inputs.begin(), IValue(grad_input_nhwc));
  AvgPool2dBackwardOutOperator::SetPTOutputs(inputs);
}

static auto& PoolKernelsKernelRegistry = habana::KernelRegistry().add(
    "aten::max_pool2d",
    KERNEL_FN(MaxPool2dOperator));
