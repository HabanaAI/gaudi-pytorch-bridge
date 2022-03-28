/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/upsample_kernels.h"

#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_lazy/lazy_executor.h"
#include "synapse_helpers/layout_utils.h"

using namespace torch;
using namespace habana;
using namespace synapse_helpers::layouts;

static bool is_tensor_5d(std::vector<int64_t> tensor_vec) {
  const uint64_t DIMS_SIZE_5 = 5;
  return tensor_vec.size() == DIMS_SIZE_5;
}

std::vector<int64_t> UpsampleOperator::compute_output_shape(
    std::vector<int64_t> shape_in,
    c10::optional<IntArrayRef> output_size,
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
    c10::optional<IntArrayRef> output_size,
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
    c10::optional<at::IntArrayRef> output_size,
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
  auto output = habana_helpers::createPTTensor(
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

void UpsampleBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto grad_output = inputs[0].toTensor();
  TORCH_CHECK(
      inputs[2].isIntList() || inputs[2].isTensor(),
      "Input 2 can be either int list or shape tensor");
  std::vector<int64_t> grad_out_shape;
  if (inputs[2].isTensor()) {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_shape_tensor());
    grad_out_shape = inputs[2].toTensor().sizes().vec();
  } else {
    grad_out_shape = inputs[2].toIntVector();
  }

  TORCH_CHECK(
      grad_output.ndimension() == 4 || grad_output.ndimension() == 5,
      "It is expected grad input tensor dimension equals to either 4 or 5, but got dim ",
      grad_output.ndimension());

  c10::optional<IntArrayRef> output_size;
  c10::optional<at::ArrayRef<double>> scales;

  auto output_size1 = inputs[1].to<c10::optional<std::vector<int64_t>>>();
  if (output_size1.has_value()) {
    output_size =
        c10::make_optional(at::ArrayRef<int64_t>(output_size1.value()));
    scales = {};
  }

  auto scales1 = inputs[3].to<c10::optional<std::vector<double>>>();
  if (scales1.has_value()) {
    scales = c10::make_optional(at::ArrayRef<double>(scales1.value()));
    output_size = {};
  }

  TORCH_CHECK(
      output_size1.has_value() || scales1.has_value(),
      "output_size and scales in UpsampleBackward are empty");

  // TPC kernel runs only ChannelLast or ChannelLast3d format
  // TPC kernel supports only 4D or 5D Tensor
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&grad_output});
  auto output = habana_helpers::createPTTensor(
      grad_output,
      grad_out_shape,
      grad_output.options(),
      memory_format,
      output_metadata.at(0).persistent);

  // Setup resize params, TF uses same
  auto is_grad_input_5d = is_tensor_5d(grad_output.sizes().vec());
  auto syn_resize_params = synapse_resize_params_builder(
      RESIZE_INTER_NEAREST,
      FLOOR,
      ASYMMETRIC_MODE,
      output_size,
      scales,
      is_grad_input_5d);

  p_context_->params_.emplace<ns_ResizeKernel::Params>(syn_resize_params);
  p_context_->params_size_ = sizeof(syn_resize_params);

  // Allocate Shape tensor
  if (graph.is_dynamic_graph() && (false == inputs[2].isTensor())) {
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

void UpsampleBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor grad_output = inputs[0].toTensor();
  auto grad_size = inputs[2].toIntList();
  std::vector<int64_t> grad_out_shape = grad_size.vec();
  auto is_grad_input_5d = is_tensor_5d(grad_output.sizes().vec());
  // Set ChannelLast for 4D Tensor or ChannelLast3d for 5D Tensor
  auto tpc_memory_format = is_grad_input_5d ? c10::MemoryFormat::ChannelsLast3d
                                            : c10::MemoryFormat::ChannelsLast;
  auto output =
      at::empty(grad_out_shape, grad_output.options(), tpc_memory_format);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

/* Mode and Alligned corners can be added here */
Tensor upsample_op_hpu(
    torch::jit::Stack& stack,
    std::string& node_type,
    UpsampleOperator* Op) {
  at::Tensor input = stack[0].toTensor();
  Tensor input_nhwc = input;
  auto is_upsample_3d = is_tensor_5d(input.sizes().vec());
  int64_t pos_in[] = {
      LayoutFormatDims::N,
      LayoutFormatDims::H,
      LayoutFormatDims::W,
      LayoutFormatDims::C};
  int64_t pos_in_3d[] = {
      LayoutFormatWithDepthDims::N,
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C};
  std::vector<const at::Tensor*> pt_in{&input};
  std::vector<at::Tensor*> pt_out{&input_nhwc};
  IntArrayRef new_dim_pos_in = pos_in;
  if (is_upsample_3d)
    new_dim_pos_in = pos_in_3d;
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Overwriting the input with the permuted input so that the inputs is in
  // channels last from this point
  stack[0] = IValue(input_nhwc);
  auto upsample_nearest = [&] {
    size_t device_id = input.device().index();
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
    size_t key = Op->GetRecipeKey(node_type, stack);
    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{input_nhwc};

    if (device.get_recipe_handle_cache().isCached(key)) {
      Op->Execute(key, pt_inputs, stack);
    } else {
      // Create Graph
      OutputMetaDataVector output_metadata(1);
      output_metadata.at(0).persistent = true;
      Op->CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
    }
    std::vector<at::Tensor> out = Op->GetOutputs();
    TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
    return out[0];
  };
  Tensor output;
  auto output_nhwc = upsample_nearest();
  pt_in = {&output_nhwc};
  pt_out = {&output};
  int64_t pos_out[] = {
      LayoutFormatDims::N,
      LayoutFormatDims::W,
      LayoutFormatDims::C,
      LayoutFormatDims::H};
  int64_t pos_out_3d[] = {
      LayoutFormatWithDepthDims::N,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C,
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H};
  IntArrayRef new_dim_pos_out = pos_out;
  if (is_upsample_3d)
    new_dim_pos_out = pos_out_3d;
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  return output;
}

Tensor upsample_backward_op_hpu(
    torch::jit::Stack& stack,
    std::string& node_type,
    UpsampleBackwardOperator* Op) {
  auto grad_output = stack[0].toTensor();
  auto input_size = stack[2].toIntList();

  Tensor grad_output_nhwc = grad_output;
  auto is_upsample_3d = is_tensor_5d(grad_output.sizes().vec());
  int64_t pos_in[] = {
      LayoutFormatDims::N,
      LayoutFormatDims::H,
      LayoutFormatDims::W,
      LayoutFormatDims::C};
  int64_t pos_in_3d[] = {
      LayoutFormatWithDepthDims::N,
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C};
  std::vector<const at::Tensor*> pt_in{&grad_output};
  std::vector<at::Tensor*> pt_out{&grad_output_nhwc};
  IntArrayRef new_dim_pos_in = pos_in;
  if (is_upsample_3d) // 5D input
    new_dim_pos_in = pos_in_3d;
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in};
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&grad_output});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  std::vector<int64_t> permuted_sizes = input_size.vec();
  permuted_sizes[0] = input_size[0];
  permuted_sizes[1] = input_size[2];
  permuted_sizes[2] = input_size[3];
  permuted_sizes[3] = input_size[1];
  if (is_upsample_3d) // 5D input
  {
    permuted_sizes[0] = input_size[0];
    permuted_sizes[1] = input_size[2];
    permuted_sizes[2] = input_size[3];
    permuted_sizes[3] = input_size[4];
    permuted_sizes[4] = input_size[1];
  }

  // Overwriting the grad_output with the permuted grad_ouput so that the inputs
  // is in channels last from this point
  stack[0] = IValue(grad_output_nhwc);
  stack[2] = IValue(permuted_sizes);
  auto upsample_nearest_backward = [&] {
    size_t device_id = grad_output.device().index();
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
    size_t key = Op->GetRecipeKey(node_type, stack);
    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{grad_output_nhwc};

    if (device.get_recipe_handle_cache().isCached(key)) {
      Op->Execute(key, pt_inputs, stack);
    } else {
      // Create Graph
      OutputMetaDataVector output_metadata(1);
      output_metadata.at(0).persistent = true;
      Op->CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
    }
    std::vector<at::Tensor> out = Op->GetOutputs();
    TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
    return out[0];
  };
  Tensor output;
  auto output_nhwc = upsample_nearest_backward();
  pt_in = {&output_nhwc};
  pt_out = {&output};
  int64_t pos_out[] = {
      LayoutFormatDims::N,
      LayoutFormatDims::W,
      LayoutFormatDims::C,
      LayoutFormatDims::H};
  int64_t pos_out_3d[] = {
      LayoutFormatWithDepthDims::N,
      LayoutFormatWithDepthDims::W,
      LayoutFormatWithDepthDims::C,
      LayoutFormatWithDepthDims::D,
      LayoutFormatWithDepthDims::H};
  IntArrayRef new_dim_pos_out = pos_out;
  if (is_upsample_3d) // 5D input
    new_dim_pos_out = pos_out_3d;
  std::vector<const IntArrayRef*> pt_new_pos1 = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos1, memory_format);

  return output;
}

Tensor upsample_nearest2d_hpu(
    const Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_KERNEL_BEGIN;
  // Create the operator
  at::ScalarType scalar_type = input.scalar_type();
  size_t device_id = input.device().index();
  habana::UpsampleNearest2dOperator Op(device_id, scalar_type);
  std::string node_type =
      "resize_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input), IValue(output_size), IValue(scale_factors)};
  auto output = upsample_op_hpu(stack, node_type, &Op);
  PT_KERNEL_END;
  return output;
}

Tensor upsample_nearest3d_hpu(
    const Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_KERNEL_BEGIN;
  // Create the operator
  at::ScalarType scalar_type = input.scalar_type();
  size_t device_id = input.device().index();
  habana::UpsampleNearest3dOperator Op(device_id, scalar_type);
  std::string node_type =
      "resize_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input), IValue(output_size), IValue(scale_factors)};
  auto output = upsample_op_hpu(stack, node_type, &Op);
  PT_KERNEL_END;
  return output;
}

Tensor upsample_nearest2d_backward_hpu(
    const Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_KERNEL_BEGIN;
  // Create the operator
  at::ScalarType scalar_type = grad_output.scalar_type();
  size_t device_id = grad_output.device().index();
  habana::UpsampleNearest2dBackwardOperator Op(device_id, scalar_type);
  std::string node_type =
      "resize_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Build Params for the graph
  // output_size and scale_factor are not necessary.
  // However added to the stack to avoid compiler unsed variable warnings
  std::vector<c10::IValue> stack = {
      IValue(grad_output),
      IValue(output_size),
      IValue(input_size),
      IValue(scale_factors)};
  auto grad_input = upsample_backward_op_hpu(stack, node_type, &Op);
  PT_KERNEL_END;
  return grad_input;
}

Tensor upsample_nearest3d_backward_hpu(
    const Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_KERNEL_BEGIN;
  // Create the operator
  at::ScalarType scalar_type = grad_output.scalar_type();
  size_t device_id = grad_output.device().index();
  habana::UpsampleNearest3dBackwardOperator Op(device_id, scalar_type);
  std::string node_type =
      "resize_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Build Params for the graph
  // output_size and scale_factor are not necessary.
  // However added to the stack to avoid compiler unsed variable warnings
  std::vector<c10::IValue> stack = {
      IValue(grad_output),
      IValue(output_size),
      IValue(input_size),
      IValue(scale_factors)};
  auto grad_input = upsample_backward_op_hpu(stack, node_type, &Op);
  PT_KERNEL_END;
  return grad_input;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::upsample_nearest2d.vec",
            KERNEL_FN(UpsampleNearest2dOperator))
        .add(
            "aten::upsample_nearest2d_backward.vec",
            KERNEL_FN(UpsampleNearest2dBackwardOperator))
        .add(
            "hpu::upsample_nearest2d_backward",
            KERNEL_FN(UpsampleNearest2dBackwardOperator))
        .add(
            "aten::upsample_nearest3d.vec",
            KERNEL_FN(UpsampleNearest3dOperator))
        .add(
            "aten::upsample_nearest3d_backward.vec",
            KERNEL_FN(UpsampleNearest3dBackwardOperator));
