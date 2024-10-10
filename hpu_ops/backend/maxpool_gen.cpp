/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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

#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "generated/backend/max_pool2d_with_indices.h"
#include "generated/backend/max_pool2d_with_indices_backward.h"
#include "generated/backend/max_pool3d_with_indices.h"
#include "generated/backend/max_pool3d_with_indices_backward.h"
#include "hpu_ops/backend/pool_helpers.h"

namespace habana {

enum MaxpoolVariant {
  MAXPOOL2D = 2,
  MAXPOOL3D = 3,
};

static int OutputShapeComputation(
    int input_shape,
    int kernel,
    int stride,
    int padding,
    int dilation,
    bool ceilMode) {
  return (
      ((input_shape + 2 * padding - dilation * (kernel - 1) - 1 +
        (ceilMode ? stride - 1 : 0)) /
       stride) +
      1);
}

OutputMetaDataVector MaxPool2DMeta(const at::Stack& stack) {
  std::vector<long int> pad = {0, 0};
  std::vector<long int> dil = {1, 1};
  auto self = stack.at(0).toTensor();
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();
  TORCH_CHECK(
      self.dim() == 4 || self.dim() == 3,
      "Maxpool2d expects Input size must be 4 or 3, but got ",
      self.dim());
  TORCH_CHECK(
      kernel.size() == 2,
      "Maxpool2d expects Kernel size must 2, but got ",
      kernel.size());
  TORCH_CHECK(
      stride.size() == 2,
      "Maxpool2d expects Stride size must 2, but got ",
      stride.size());
  TORCH_CHECK(
      padding.size() == 2,
      "Maxpool2d expects Padding size must 2, but got ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 2,
      "Maxpool2d expects Dilation size must 2, but got ",
      dilation.size());
  std::vector<int64_t> input_shape = self.sizes().vec();
  std::vector<int64_t> output_shape = self.sizes().vec();

  int n = kernel.size();
  // updating the width & height dimension
  int output_shape_index = output_shape.size() - n;
  for (int i = 0; i < n; i++) {
    output_shape.at(output_shape_index) = OutputShapeComputation(
        input_shape.at(output_shape_index),
        kernel[i],
        stride[i],
        padding[i],
        dilation[i],
        ceil_mode);
    output_shape_index++;
  }

  // ensure that the last pooling starts inside the image
  // needed to avoid problems in ceil mode
  if (ceil_mode) {
    for (int i = 0; i < n; i++) {
      if ((output_shape.rbegin()[i] - 1) * stride[n - i - 1] >=
          input_shape.rbegin()[i] + padding[n - i - 1])
        --output_shape.rbegin()[i];
    }
  }

  OutputMetaDataVector meta;
  meta.resize(2);

  meta[0].shape = output_shape;
  meta[0].dtype = self.scalar_type();

  meta[1].shape = output_shape;
  meta[1].dtype = at::kLong;

  return meta;
}

OutputMetaDataVector MaxPoolMetaBwd(const at::Stack& stack) {
  auto self = stack.at(1).toTensor();
  at::Stack stack_fwd(stack.begin() + 1, stack.end());
  auto grad = stack.at(0).toTensor();
  auto kernel = stack.at(2).toIntVector();
  std::vector<int64_t> indices;
  if (kernel.size() == 2) {
    indices = MaxPool2DMeta(stack_fwd)[0].shape;
  }
  if (kernel.size() == 3) {
    indices = Maxpool3dWithIndicesMeta(stack_fwd)[0].shape;
  }
  HABANA_ASSERT(
      (grad.sizes() == indices), "Grad and Indices sizes don't match");

  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();

  return {meta};
}

sizes_vec MaxPool3DIndicesOutputShape(const at::Stack& stack) {
  std::vector<long int> pad = {0, 0, 0};
  std::vector<long int> dil = {1, 1, 1};
  auto self = stack.at(0).toTensor();
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();

  TORCH_CHECK(
      self.dim() == 5 || self.dim() == 4,
      "Maxpool3d expects Input size must be 5 or 4, but got ",
      self.dim());
  TORCH_CHECK(
      padding.size() == 3,
      "Maxpool3d expects padding size is 3 but got ",
      padding.size());
  TORCH_CHECK(
      kernel.size() == 3,
      "Maxpool3d expects kernel size is 3 but got ",
      kernel.size());
  TORCH_CHECK(
      stride.size() == 3,
      "Maxpool3d expects stride size is 3 but got ",
      stride.size());
  TORCH_CHECK(
      dilation.size() == 3,
      "Maxpool3d expects dilation size is 3 but got ",
      dilation.size());

  std::vector<int64_t> input_shape = self.sizes().vec();
  std::vector<int64_t> output_shape = self.sizes().vec();

  int n = kernel.size();
  int output_shape_index = output_shape.size() - n;
  // updating the width, height, & depth dimension
  for (int i = 0; i < n; i++) {
    output_shape.at(output_shape_index) = OutputShapeComputation(
        input_shape.at(output_shape_index),
        kernel[i],
        stride[i],
        padding[i],
        dilation[i],
        ceil_mode);
    output_shape_index++;
  }

  // ensure that the last pooling starts inside the image
  // needed to avoid problems in ceil mode
  if (ceil_mode) {
    int index_ceil_mode = output_shape.size() - n;
    for (int i = 0; i < n; i++) {
      if ((output_shape.at(index_ceil_mode) - 1) * stride[i] >=
          input_shape.at(index_ceil_mode) + padding[i])
        --output_shape.at(index_ceil_mode);
      index_ceil_mode++;
    }
  }
  return {output_shape, output_shape};
}

OutputMetaDataVector Maxpool3dWithIndicesMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto& output_shape = MaxPool3DIndicesOutputShape(stack);
  OutputMetaDataVector meta;
  meta.resize(output_shape.size());

  meta[0].shape = output_shape[0];
  meta[0].dtype = self.scalar_type();

  meta[1].shape = output_shape[0];
  meta[1].dtype = c10::ScalarType::Long;

  return meta;
}

static std::shared_ptr<void> FillSpatialReduction3DParams(
    std::vector<int64_t>& kernel,
    std::vector<int64_t>& stride,
    std::vector<int64_t>& padding,
    std::vector<int64_t>& dilation,
    bool ceil_mode,
    size_t& size) {
  PARAMS_STUB(ns_SpatialReduction3D::Params);
  params->pad_w_begin = padding[2];
  params->pad_w_end = padding[2];
  params->pad_h_begin = padding[1];
  params->pad_h_end = padding[1];
  params->pad_d_begin = padding[0];
  params->pad_d_end = padding[0];
  params->kernel_w = kernel[2];
  params->kernel_h = kernel[1];
  params->kernel_d = kernel[0];
  params->stride_w = stride[2];
  params->stride_h = stride[1];
  params->stride_d = stride[0];
  params->dilation_w = dilation[2];
  params->dilation_h = dilation[1];
  params->dilation_d = dilation[0];
  if (ceil_mode)
    params->pooling_convention =
        EPoolingConvention::POOLING_CONVENTION_FULL_PYTORCH;
  else
    params->pooling_convention = EPoolingConvention::POOLING_CONVENTION_VALID;
  return params;
}

std::shared_ptr<void> FillSpatialReduction3DParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0, 0};
  std::vector<long int> dil = {1, 1, 1};
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();

  return FillSpatialReduction3DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

std::shared_ptr<void> FillSpatialReduction3DParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0, 0};
  std::vector<long int> dil = {1, 1, 1};
  auto kernel = stack.at(2).toIntVector();
  auto stride = stack.at(3).toIntVector().size() == 0
      ? kernel
      : stack.at(3).toIntVector();
  auto padding =
      stack.at(4).toIntVector().size() == 0 ? pad : stack.at(4).toIntVector();
  auto dilation =
      stack.at(5).toIntVector().size() == 0 ? dil : stack.at(5).toIntVector();
  const bool ceil_mode = stack.at(6).toBool();

  return FillSpatialReduction3DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

static std::shared_ptr<void> FillSpatialReduction2DParams(
    std::vector<int64_t>& kernel,
    std::vector<int64_t>& stride,
    std::vector<int64_t>& padding,
    std::vector<int64_t>& dilation,
    bool ceil_mode,
    size_t& size) {
  PARAMS_STUB(ns_SpatialReduction::Params);
  params->pad_w_begin = padding[1];
  params->pad_w_end = padding[1];
  params->pad_h_begin = padding[0];
  params->pad_h_end = padding[0];
  params->kernel_w = kernel[1];
  params->kernel_h = kernel[0];
  params->stride_w = stride[1];
  params->stride_h = stride[0];
  params->dilation_w = dilation[1];
  params->dilation_h = dilation[0];
  if (ceil_mode)
    params->pooling_convention =
        EPoolingConvention::POOLING_CONVENTION_FULL_PYTORCH;
  else
    params->pooling_convention = EPoolingConvention::POOLING_CONVENTION_VALID;
  return params;
}

std::shared_ptr<void> FillSpatialReduction2DParamsFwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0};
  std::vector<long int> dil = {1, 1};
  auto kernel = stack.at(1).toIntVector();
  auto stride = stack.at(2).toIntVector().size() == 0
      ? kernel
      : stack.at(2).toIntVector();
  auto padding =
      stack.at(3).toIntVector().size() == 0 ? pad : stack.at(3).toIntVector();
  auto dilation =
      stack.at(4).toIntVector().size() == 0 ? dil : stack.at(4).toIntVector();
  const bool ceil_mode = stack.at(5).toBool();

  return FillSpatialReduction2DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

std::shared_ptr<void> FillSpatialReduction2DParamsBwd(
    const at::Stack& stack,
    size_t& size) {
  std::vector<long int> pad = {0, 0};
  std::vector<long int> dil = {1, 1};
  auto kernel = stack.at(2).toIntVector();
  auto stride = stack.at(3).toIntVector().size() == 0
      ? kernel
      : stack.at(3).toIntVector();
  auto padding =
      stack.at(4).toIntVector().size() == 0 ? pad : stack.at(4).toIntVector();
  auto dilation =
      stack.at(5).toIntVector().size() == 0 ? dil : stack.at(5).toIntVector();
  const bool ceil_mode = stack.at(6).toBool();

  return FillSpatialReduction2DParams(
      kernel, stride, padding, dilation, ceil_mode, size);
}

static at::ScalarType FindRetainTensorType(at::ScalarType inputTensorType) {
  switch (inputTensorType) {
    case at::ScalarType::BFloat16:
    case at::ScalarType::Half:
      return at::ScalarType::Short;
    default:
      return at::ScalarType::Byte;
  }
}

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263)
void MaxPool3DWithIndicesOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  std::vector<synTensor> inputs = {syn_in(0)};
  const auto meta = Maxpool3dWithIndicesMeta(stack)[0];
  size_t size = 0;
  auto index_type = FindRetainTensorType(meta.dtype);
  const auto& params = FillSpatialReduction3DParamsFwd(stack, size);
  const auto rank = stack_tensor(stack, 0).dim();

  if (rank == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDC},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDC});
  } else if (rank == 5) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }

  auto maxpool3d = BuildOp(
      graph,
      GetGuid(),
      std::move(inputs),
      {{meta.shape, index_type}, {meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(maxpool3d[1]);
  syn_out(1) = BuildCast(
      this, graph, maxpool3d.at(0).get(), meta.shape, index_type, at::kLong, 1);
}

void MaxPool3DWithIndicesBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = MaxPoolMetaBwd(stack)[0];
  size_t size = 0;
  const auto params = FillSpatialReduction3DParamsBwd(stack, size);

  auto cast_input = BuildCast(
      this,
      graph,
      syn_in(2),
      stack.back().toTensor().sizes(),
      at::kLong,
      FindRetainTensorType(meta.dtype));

  std::vector<synTensor> inputs = {syn_in(0), cast_input.get()};
  const auto rank = stack_tensor(stack, 0).dim();

  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  if (rank == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDC,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDC},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDC});
  } else if (rank == 5) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHDCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHDCN});
  }

  auto grad_output = BuildOp(
      graph,
      GetGuid(),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(grad_output[0]);
}

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263)
void MaxPool2DWithIndices::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = MaxPool2DMeta(stack)[0];
  size_t size = 0;
  const auto& params = FillSpatialReduction2DParamsFwd(stack, size);

  if (stack_tensor(stack, 0).dim() == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  } else {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHN});
  }

  auto maxPool2d = BuildOp(
      graph,
      GetGuid(),
      {syn_in(0)},
      {{meta.shape, meta.dtype, 0}, {meta.shape, at::kLong, 1}},
      params.get(),
      size);
  syn_out(0) = std::move(maxPool2d[0]);
  syn_out(1) = std::move(maxPool2d[1]);
}

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263)
void MaxPool2DWithIndicesBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = MaxPoolMetaBwd(stack).at(0);
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  const auto& inputDimensions = stack_tensor(stack, 0).dim();

  if (inputDimensions == 4) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  } else if (inputDimensions == 3) {
    SetSynapseLayouts(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHN},
        {synapse_helpers::layouts::SynapseLayoutFormat::WHN});
  }

  auto maxpool2d_gradout = BuildOp(
      graph,
      GetGuid(),
      {syn_in(0), syn_in(1), syn_in(2)},
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(maxpool2d_gradout.at(0));
}
} // namespace habana
