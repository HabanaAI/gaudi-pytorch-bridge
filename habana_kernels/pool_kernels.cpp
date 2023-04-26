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
#include "habana_helpers/logging.h"
#include "habana_kernels/conv_pool_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/tensor_impl.h"
#include "hpu_ops/backend/pool_helpers.h"

namespace habana {
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

OutputShapeInfRetType MaxPool2dOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  const auto dilation = inputs[4].toIntList().vec();
  bool ceil_mode = inputs[5].toBool();

  auto shape_out = compute_pool_kernel_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode);

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

void MaxPool2dOperator::AllocateAndAddSynapseNode(
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

  auto out_shape = compute_pool_kernel_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode);

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

void MaxPool2dOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  const auto dilation = inputs[4].toIntList().vec();
  bool ceil_mode = inputs[5].toBool();

  auto out_shape = compute_pool_kernel_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode);

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
