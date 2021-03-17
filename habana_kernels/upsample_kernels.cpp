/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/upsample_kernels.h"

using namespace torch;
using namespace habana;
std::vector<int64_t> UpsampleOperator::compute_output_shape(
    std::vector<int64_t> shape_in,
    c10::optional<IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scales,
    c10::MemoryFormat memory_format) {
  HABANA_ASSERT(
      (memory_format == c10::MemoryFormat::ChannelsLast) ||
      (memory_format == c10::MemoryFormat::Contiguous));
  HABANA_ASSERT(scales.has_value() || output_size.has_value());

  std::vector<int64_t> out_shape;
  if (scales.has_value()) {
    auto scale_factor = scales.value().vec();
    if (memory_format == c10::MemoryFormat::ChannelsLast)
      out_shape = {
          shape_in[0],
          static_cast<int64_t>(shape_in[1] * scale_factor[0]),
          static_cast<int64_t>(shape_in[2] * scale_factor[1]),
          shape_in[3]};
    else
      out_shape = {
          shape_in[0],
          shape_in[1],
          static_cast<int64_t>(shape_in[2] * scale_factor[0]),
          static_cast<int64_t>(shape_in[3] * scale_factor[1])};
  } else {
    auto out_size = output_size.value().vec();
    if (memory_format == c10::MemoryFormat::ChannelsLast)
      out_shape = {shape_in[0], out_size[0], out_size[1], shape_in[3]};
    else
      out_shape = {shape_in[0], shape_in[1], out_size[0], out_size[1]};
  }
  return out_shape;
}

/**
 * @brief Fill generic upsample params structure
 */
ns_UpsampleKernel::Params synapse_upsample_params_builder(
    EUpsampleType_t mode,
    int scale) {
  ns_UpsampleKernel::Params upsample_params{};
  upsample_params.mode = mode;
  upsample_params.scale = scale;

  return upsample_params;
}

void UpsampleOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  auto input = inputs[0].toTensor();
  auto output_size = inputs[1].toOptionalIntArray();
  auto scales = inputs[2].toOptionalDoubleArray();
  TORCH_CHECK(
      input.ndimension() == 4,
      "It is expected input tensor dimension equals to 4, but got size ",
      input.ndimension());

  // TPC kernel runs only ChannelLast format
  // TPC kernel supports only 4D Tensor
  std::vector<int64_t> shape_out = compute_output_shape(
      input.sizes().vec(),
      output_size,
      scales,
      c10::MemoryFormat::ChannelsLast);

  auto output = habana_helpers::createPTTensor(
      input,
      shape_out,
      input.options(),
      c10::MemoryFormat::ChannelsLast,
      is_output_persistent);

  // Setup pool params
  auto syn_upsample_params =
      synapse_upsample_params_builder(UPSAMPLE_TYPE_NEAREST_NEIGHBOR, 1);

  p_context_->params_.emplace<ns_UpsampleKernel::Params>(syn_upsample_params);
  p_context_->params_size_ = sizeof(syn_upsample_params);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(
      graph, &syn_upsample_params, sizeof(syn_upsample_params));
}

void UpsampleBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  PT_KERNEL_BEGIN;
  auto grad_output = inputs[0].toTensor();
  auto grad_size = inputs[2].toIntList();
  std::vector<int64_t> grad_out_shape = grad_size.vec();
  TORCH_CHECK(
      grad_output.ndimension() == 4,
      "It is expected grad input tensor dimension equals to 4, but got size ",
      grad_output.ndimension());
  // TPC kernel runs only ChannelLast format
  // TPC kernel supports only 4D Tensor
  auto output = habana_helpers::createPTTensor(
      grad_output,
      grad_out_shape,
      grad_output.options(),
      c10::MemoryFormat::ChannelsLast,
      is_output_persistent);

  // Setup upsample params
  auto syn_upsample_params =
      synapse_upsample_params_builder(UPSAMPLE_TYPE_NEAREST_NEIGHBOR, 1);

  p_context_->params_.emplace<ns_UpsampleKernel::Params>(syn_upsample_params);
  p_context_->params_size_ = sizeof(syn_upsample_params);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(
      graph, &syn_upsample_params, sizeof(syn_upsample_params));
  PT_KERNEL_END;
}

void UpsampleOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  auto output_size = inputs[1].toOptionalIntArray();
  auto scales = inputs[2].toOptionalDoubleArray();
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});

  std::vector<int64_t> shape_out = compute_output_shape(
      input.sizes().vec(),
      output_size,
      scales,
      c10::MemoryFormat::ChannelsLast);
  auto output = at::empty(shape_out, input.options(), memory_format);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

void UpsampleBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor grad_output = inputs[0].toTensor();
  auto grad_size = inputs[2].toIntList();
  std::vector<int64_t> grad_out_shape = grad_size.vec();
  auto output = at::empty(
      grad_out_shape, grad_output.options(), c10::MemoryFormat::ChannelsLast);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

/* Mode and Alligned corners can be added here */
Tensor upsample_op_hpu(
    torch::jit::Stack& stack,
    std::string& node_type,
    UpsampleOperator* Op) {
  PT_KERNEL_BEGIN;
  at::Tensor input = stack[0].toTensor();
  Tensor input_nhwc = input;
  int64_t pos_in[] = {0, 2, 3, 1};
  std::vector<const at::Tensor*> pt_in{&input};
  std::vector<at::Tensor*> pt_out{&input_nhwc};
  IntArrayRef new_dim_pos_in = pos_in;
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Overwriting the input with the permuted input so that the inputs is in
  // channels last from this point
  stack[0] = IValue(input_nhwc);
  auto upsample_nearest2d = [&] {
    size_t device_id = input.device().index();
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
    size_t key = Op->GetRecipeKey(node_type, stack);
    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{input_nhwc};

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op->SetPTInputs(pt_inputs);
      Op->SetPTOutputs(stack);
      Op->Execute(key);
    } else {
      PT_KERNEL_DEBUG("key:", key);
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      Op->AllocateSynapseInputs(graph, pt_inputs, true);
      Op->AllocateAndAddSynapseNode(graph, stack, true);
      Op->Compile(graph);
    }
    std::vector<at::Tensor> out = Op->GetOutputs();
    TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
    return out[0];
  };
  Tensor output;
  auto output_nhwc = upsample_nearest2d();
  pt_in = {&output_nhwc};
  pt_out = {&output};
  int64_t pos_out[] = {0, 3, 1, 2};
  IntArrayRef new_dim_pos_out = pos_out;
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  PT_KERNEL_END;
  return output;
}

Tensor upsample_backward_op_hpu(
    torch::jit::Stack& stack,
    std::string& node_type,
    UpsampleBackwardOperator* Op) {
  PT_KERNEL_BEGIN;
  auto grad_output = stack[0].toTensor();
  auto input_size = stack[2].toIntList();

  Tensor grad_output_nhwc = grad_output;
  int64_t pos_in[] = {0, 2, 3, 1};
  std::vector<const at::Tensor*> pt_in{&grad_output};
  std::vector<at::Tensor*> pt_out{&grad_output_nhwc};
  IntArrayRef new_dim_pos_in = pos_in;
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in};
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&grad_output});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  std::vector<int64_t> permuted_sizes = input_size.vec();
  if (memory_format == c10::MemoryFormat::Contiguous) {
    permuted_sizes[0] = input_size[0];
    permuted_sizes[1] = input_size[2];
    permuted_sizes[2] = input_size[3];
    permuted_sizes[3] = input_size[1];
  }
  // Overwriting the grad_output with the permuted grad_ouput so that the inputs
  // is in channels last from this point
  stack[0] = IValue(grad_output_nhwc);
  stack[2] = IValue(permuted_sizes);
  auto upsample_nearest2d_backward = [&] {
    size_t device_id = grad_output.device().index();
    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
    size_t key = Op->GetRecipeKey(node_type, stack);
    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{grad_output_nhwc};

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op->SetPTInputs(pt_inputs);
      Op->SetPTOutputs(stack);
      Op->Execute(key);
    } else {
      PT_KERNEL_DEBUG("key:", key);
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      Op->AllocateSynapseInputs(graph, pt_inputs, true);
      Op->AllocateAndAddSynapseNode(graph, stack, true);
      Op->Compile(graph);
    }
    std::vector<at::Tensor> out = Op->GetOutputs();
    TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
    return out[0];
  };
  Tensor output;
  auto output_nhwc = upsample_nearest2d_backward();
  pt_in = {&output_nhwc};
  pt_out = {&output};
  int64_t pos_out[] = {0, 3, 1, 2};
  IntArrayRef new_dim_pos_out = pos_out;
  std::vector<const IntArrayRef*> pt_new_pos1 = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos1, memory_format);

  PT_KERNEL_END;
  return output;
}

Tensor upsample_nearest2d_hpu(
    const Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  // Create the operator
  at::ScalarType scalar_type = input.scalar_type();
  size_t device_id = input.device().index();
  habana::UpsampleNearest2dOperator Op(device_id, scalar_type);
  std::string node_type =
      "upsample_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(input), IValue(output_size), IValue(scale_factors)};
  return upsample_op_hpu(stack, node_type, &Op);
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
      "upsample_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Build Params for the graph
  // output_size and scale_factor are not necessary.
  // However added to the stack to avoid compiler unsed variable warnings
  std::vector<c10::IValue> stack = {
      IValue(grad_output),
      IValue(output_size),
      IValue(input_size),
      IValue(scale_factors)};
  PT_KERNEL_END;
  return upsample_backward_op_hpu(stack, node_type, &Op);
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::upsample_nearest2d",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<UpsampleNearest2dOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::upsample_nearest2d_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<UpsampleNearest2dBackwardOperator>(
                  device_id, node_type);
            });
