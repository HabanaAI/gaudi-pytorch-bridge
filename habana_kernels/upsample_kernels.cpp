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
#include "habana_kernels/upsample_kernels.h"

#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_lazy/lazy_executor.h"

using namespace torch;
using namespace habana;
using namespace synapse_helpers::layouts;

static bool is_tensor_5d(std::vector<int64_t> tensor_vec) {
  const uint64_t DIMS_SIZE_5 = 5;
  return tensor_vec.size() == DIMS_SIZE_5;
}

std::vector<int64_t> UpsampleOperator::compute_output_shape(
    std::vector<int64_t> shape_in,
    at::OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scales) {
  TORCH_CHECK(
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING),
      "compute_output_shape for Synapse layout handling mode");
  TORCH_CHECK(
      scales.has_value() || output_size.has_value(),
      "Either Upsample scales or output_size not defined");
  std::vector<int64_t> out_shape;
  bool is_input_5d = is_tensor_5d(shape_in);
  if (scales.has_value()) {
    auto scale_factor_in_double = scales.value().vec();
    // Cast scale_factor from double -> float. This is required so that output
    // shape computed matches OFM computation in TPC Glue code for resize
    // kernel.
    std::vector<float> scale_factor(
        scale_factor_in_double.begin(), scale_factor_in_double.end());
    if (is_input_5d) { // Upsample nearest 3d
      out_shape = {
          shape_in[INPUT_3D_N_IDX],
          shape_in[INPUT_3D_C_IDX],
          static_cast<int64_t>(shape_in[INPUT_3D_D_IDX] * scale_factor[0]),
          static_cast<int64_t>(shape_in[INPUT_3D_H_IDX] * scale_factor[1]),
          static_cast<int64_t>(shape_in[INPUT_3D_W_IDX] * scale_factor[2])};
    } else { // Upsample nearest 2d
      out_shape = {
          shape_in[INPUT_N_IDX],
          shape_in[INPUT_C_IDX],
          static_cast<int64_t>(shape_in[INPUT_H_IDX] * scale_factor[0]),
          static_cast<int64_t>(shape_in[INPUT_W_IDX] * scale_factor[1])};
    }
  } else if (output_size.has_value()) {
    auto out_size = output_size.value().vec();
    if (is_input_5d) { // Upsample nearest 3d
      out_shape = {
          shape_in[INPUT_3D_N_IDX],
          shape_in[INPUT_3D_C_IDX],
          out_size[0],
          out_size[1],
          out_size[2]};
    } else { // Upsample nearest 2d
      out_shape = {
          shape_in[INPUT_N_IDX],
          shape_in[INPUT_C_IDX],
          out_size[0],
          out_size[1]};
    }
  }
  return out_shape;
}

std::vector<int64_t> UpsampleOperator::compute_output_shape(
    std::vector<int64_t> shape_in,
    at::OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scales,
    c10::MemoryFormat memory_format) {
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING) == true) {
    return compute_output_shape(shape_in, output_size, scales);
  }
  TORCH_CHECK(
      (memory_format == c10::MemoryFormat::ChannelsLast3d) ||
          (memory_format == c10::MemoryFormat::ChannelsLast) ||
          (memory_format == c10::MemoryFormat::Contiguous),
      "Unsupported Upsample memory format ",
      memory_format);
  TORCH_CHECK(
      scales.has_value() || output_size.has_value(),
      "Either Upsample scales or output_size not defined");
  std::vector<int64_t> out_shape;
  bool is_input_5d = is_tensor_5d(shape_in);
  if (scales.has_value()) {
    auto isLowering = habana_lazy::isDeviceInLoweringMode();

    auto scale_factor_in_double = scales.value().vec();
    // Cast scale_factor from double -> float. This is required so that output
    // shape computed matches OFM computation in TPC Glue code for resize
    // kernel.
    std::vector<float> scale_factor(
        scale_factor_in_double.begin(), scale_factor_in_double.end());
    if (is_input_5d) { // Upsample nearest 3d
      TORCH_CHECK(
          memory_format != c10::MemoryFormat::ChannelsLast,
          "Upsample_nearest3d input called with memory format ChannelsLast");
      if (memory_format == c10::MemoryFormat::ChannelsLast3d &&
          isLowering) // Layout NDHWC
        out_shape = {
            shape_in[0],
            static_cast<int64_t>(shape_in[1] * scale_factor[0]),
            static_cast<int64_t>(shape_in[2] * scale_factor[1]),
            static_cast<int64_t>(shape_in[3] * scale_factor[2]),
            shape_in[4]};
      else // Layout NCDHW
        out_shape = {
            shape_in[0],
            shape_in[1],
            static_cast<int64_t>(shape_in[2] * scale_factor[0]),
            static_cast<int64_t>(shape_in[3] * scale_factor[1]),
            static_cast<int64_t>(shape_in[4] * scale_factor[2])};
    } else { // Upsample nearest 2d
      TORCH_CHECK(
          memory_format != c10::MemoryFormat::ChannelsLast3d,
          "Upsample_nearest2d input called with memory format ChannelsLast3d");
      if (memory_format == c10::MemoryFormat::ChannelsLast &&
          isLowering) // Layout NHWC
        out_shape = {
            shape_in[0],
            static_cast<int64_t>(shape_in[1] * scale_factor[0]),
            static_cast<int64_t>(shape_in[2] * scale_factor[1]),
            shape_in[3]};
      else // Layout NCHW
        out_shape = {
            shape_in[0],
            shape_in[1],
            static_cast<int64_t>(shape_in[2] * scale_factor[0]),
            static_cast<int64_t>(shape_in[3] * scale_factor[1])};
    }
  } else if (output_size.has_value()) {
    auto out_size = output_size.value().vec();
    if (is_input_5d) { // Upsample nearest 3d
      TORCH_CHECK(
          memory_format != c10::MemoryFormat::ChannelsLast,
          "Upsample_nearest3d input called with memory format ChannelsLast");
      if (memory_format == c10::MemoryFormat::ChannelsLast3d)
        out_shape = {
            shape_in[0], out_size[0], out_size[1], out_size[2], shape_in[4]};
      else
        out_shape = {
            shape_in[0], shape_in[1], out_size[0], out_size[1], out_size[2]};
    } else { // Upsample nearest 2d
      TORCH_CHECK(
          memory_format != c10::MemoryFormat::ChannelsLast3d,
          "Upsample_nearest2d input called with memory format ChannelsLast3d");
      if (memory_format == c10::MemoryFormat::ChannelsLast)
        out_shape = {shape_in[0], out_size[0], out_size[1], shape_in[3]};
      else
        out_shape = {shape_in[0], shape_in[1], out_size[0], out_size[1]};
    }
  } else {
    TORCH_CHECK(0, "Upsample_nearest2d/3d called without scales or out_size");
  }
  return out_shape;
}

/**
 * @brief Fill generic resize params structure
 */
ns_ResizeKernel::Params synapse_resize_params_builder(
    ResizeInterpolationMode_t interp_mode,
    ResizeNearestMode_t nearest_modetype,
    ResizeCoordinateTransformationMode_t coord_mode,
    OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors,
    const bool is_upsample_3d = false) {
  ns_ResizeKernel::Params resize_params{};
  resize_params.mode = interp_mode;
  resize_params.coordTransMode = coord_mode;
  resize_params.nearestMode = nearest_modetype;
  resize_params.useScales = scale_factors.has_value();
  resize_params.excludeOutside = false;
  // resize_params.cubicCoeffA = NA;
  if (is_upsample_3d) {
    if (resize_params.useScales) {
      resize_params.scaleDim1 = scale_factors.value()[2];
      resize_params.scaleDim2 = scale_factors.value()[1];
      resize_params.scaleDim3 = scale_factors.value()[0];
    } else {
      resize_params.size1 = output_size.value()[2];
      resize_params.size2 = output_size.value()[1];
      resize_params.size3 = output_size.value()[0];
    }
  } else {
    if (resize_params.useScales) {
      resize_params.scaleDim1 = scale_factors.value()[1];
      resize_params.scaleDim2 = scale_factors.value()[0];
      resize_params.scaleDim3 = 1.0;
    } else {
      resize_params.size1 = output_size.value()[1];
      resize_params.size2 = output_size.value()[0];
      resize_params.size3 = 1;
    }
  }

  return resize_params;
}

OutputShapeInfRetType UpsampleOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto input = inputs[0].toTensor();

  c10::optional<IntArrayRef> output_size;
  c10::optional<at::ArrayRef<double>> scales;
  auto output_size1 = inputs[1].to<c10::optional<std::vector<int64_t>>>();
  output_size = output_size1.has_value()
      ? c10::make_optional(ArrayRef<int64_t>(output_size1.value()))
      : c10::nullopt;
  auto scales1 = inputs[2].to<c10::optional<std::vector<double>>>();
  scales = scales1.has_value()
      ? c10::make_optional(ArrayRef<double>(scales1.value()))
      : c10::nullopt;

  // TPC kernel runs only ChannelLast or ChannelLast3d format
  // TPC kernel supports only 4D or 5D Tensor
  auto is_input_5d = is_tensor_5d(input.sizes().vec());
  // Set ChannelLast for 4D Tensor or ChannelLast 5D Tensor
  auto tpc_memory_format = is_input_5d ? c10::MemoryFormat::ChannelsLast3d
                                       : c10::MemoryFormat::ChannelsLast;
  std::vector<int64_t> shape_out = compute_output_shape(
      input.sizes().vec(), output_size, scales, tpc_memory_format);

  OutputShapeInfRetType out;
  auto tensor_meta_data = TensorMetaData(
      shape_out,
      HabanaOperator::CalculateStrides(
          shape_out, input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format());
  out.AddOutputTensor(tensor_meta_data);
  out.AddShapeTensor(tensor_meta_data);

  return out;
}

void UpsampleOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto input = inputs[0].toTensor();

  TORCH_CHECK(
      input.ndimension() == 4 || input.ndimension() == 5,
      "It is expected input tensor dimension equals to either 4 or 5, but got dim ",
      input.ndimension());

  c10::optional<IntArrayRef> output_size;
  c10::optional<at::ArrayRef<double>> scales;
  auto output_size1 = inputs[1].to<c10::optional<std::vector<int64_t>>>();
  output_size = output_size1.has_value()
      ? c10::make_optional(ArrayRef<int64_t>(output_size1.value()))
      : c10::nullopt;
  // toOptionalIntArray and toOptionalDoubleArray are deprecated

  auto scales1 = inputs[2].to<c10::optional<std::vector<double>>>();
  scales = scales1.has_value()
      ? c10::make_optional(ArrayRef<double>(scales1.value()))
      : c10::nullopt;

  TORCH_CHECK(
      output_size1.has_value() || scales1.has_value(),
      "output_size and scales in Upsample are empty");

  // TPC kernel runs only ChannelLast or ChannelLast3d format
  // TPC kernel supports only 4D or 5D Tensor
  auto is_input_5d = is_tensor_5d(input.sizes().vec());
  // Set ChannelLast for 4D Tensor or ChannelLast 5D Tensor
  auto tpc_memory_format = is_input_5d ? c10::MemoryFormat::ChannelsLast3d
                                       : c10::MemoryFormat::ChannelsLast;
  std::vector<int64_t> shape_out = compute_output_shape(
      input.sizes().vec(), output_size, scales, tpc_memory_format);
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  auto output = habana::createPTTensor(
      input,
      shape_out,
      input.options(),
      memory_format,
      output_metadata.at(0).persistent);

  // Setup resize params, TF uses same
  auto syn_resize_params = synapse_resize_params_builder(
      RESIZE_INTER_NEAREST,
      FLOOR,
      ASYMMETRIC_MODE,
      output_size,
      scales,
      is_input_5d);

  p_context_->params_.emplace<ns_ResizeKernel::Params>(syn_resize_params);
  p_context_->params_size_ = sizeof(syn_resize_params);

  // Allocate Shape tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &syn_resize_params, sizeof(syn_resize_params));
}

void UpsampleOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  auto output_size1 = inputs[1].to<c10::optional<std::vector<int64_t>>>();
  auto output_size = output_size1.has_value()
      ? c10::make_optional(ArrayRef<int64_t>(output_size1.value()))
      : c10::nullopt;
  auto scales1 = inputs[2].to<c10::optional<std::vector<double>>>();
  auto scales = scales1.has_value()
      ? c10::make_optional(ArrayRef<double>(scales1.value()))
      : c10::nullopt;

  auto is_input_5d = is_tensor_5d(input.sizes().vec());
  // Set ChannelLast for 4D Tensor or ChannelLast3d for 5D Tensor
  auto tpc_memory_format = is_input_5d ? c10::MemoryFormat::ChannelsLast3d
                                       : c10::MemoryFormat::ChannelsLast;
  std::vector<int64_t> shape_out = compute_output_shape(
      input.sizes().vec(), output_size, scales, tpc_memory_format);
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  auto output = at::empty(shape_out, input.options(), memory_format);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}
