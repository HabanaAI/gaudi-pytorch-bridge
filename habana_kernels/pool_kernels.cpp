/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/InferSize.h>
// #include <ATen/native/Pool.h> // TODO: fix this include
#include <ATen/div_rtn.h> // TODO: remove this header after ATen/native/Pool.h is included
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <algorithm>
#include <iostream>

#include "habana_device/HPUCheck.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/conv_pool_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

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

/**
 * @brief Compute shape for output tensor(s) from given input tensor shape
 *         & pooling params such as kernel, stride, pad, dilation, ceil_mode
 */
static std::vector<int64_t> compute_output_shape(
    const at::Tensor& input,
    const at::IntArrayRef kernel_size,
    const at::IntArrayRef stride,
    const at::IntArrayRef padding,
    const at::IntArrayRef dilation,
    bool ceil_mode,
    bool is_input_nhwc = false) {
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

} // namespace

/**
 * @brief Fill generic pooling params structure
 */
ns_SpatialReduction::Params synapse_pool_params_builder(
    const IntArrayRef& kernel_size, // HW
    const IntArrayRef& stride, // HW
    const IntArrayRef& padding, // HW
    const IntArrayRef& dilation // HW
) {
  const int64_t filter_H = kernel_size[0];
  const int64_t filter_W = kernel_size[1];
  const int64_t stride_H = stride[0];
  const int64_t stride_W = stride[1];
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
  pool_params.pooling_convention = POOLING_CONVENTION_VALID;

  return pool_params;
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
  *pt_pool_params =
      synapse_pool_params_builder(kernel_size, stride, padding, dilation);
  avg_pool_params.includePadding = include_padding;

  return avg_pool_params;
}

void MaxPool2dWithIndicesOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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

  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  const auto dilation = inputs[4].toIntList().vec();
  bool ceil_mode = inputs[5].toBool();

  // Setup pool params
  auto syn_pool_params =
      synapse_pool_params_builder(kernel_size, stride, padding, dilation);

  p_context_->params_.emplace<ns_SpatialReduction::Params>(syn_pool_params);
  p_context_->params_size_ = sizeof(syn_pool_params);

  auto out_shape = compute_output_shape(
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

  AllocateSynapseOutputs(
      graph, {output_idx_nhwc, output_nhwc}, is_output_persistent);
  AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
  std::swap(p_context_->pt_outputs_[0], p_context_->pt_outputs_[1]);
  std::swap(p_context_->syn_outputs_[0], p_context_->syn_outputs_[1]);
}

void MaxPool2dWithIndicesBackwardOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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
  auto syn_pool_params =
      synapse_pool_params_builder(kernel_size, stride, padding, dilation);

  p_context_->params_.emplace<ns_SpatialReduction::Params>(syn_pool_params);
  p_context_->params_size_ = sizeof(syn_pool_params);

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[1], out_shape[2], out_shape[3]};
  TORCH_CHECK(grad_out.sizes().vec() == expected_output_size);
  TORCH_CHECK(input.sizes() == grad_input.sizes());
  TORCH_CHECK(grad_out.sizes() == indices.sizes());

  TORCH_CHECK(
      (indices.scalar_type() == c10::ScalarType::Byte) ||
      (indices.scalar_type() == c10::ScalarType::Short));

  AllocateSynapseOutput(graph, {grad_input}, is_output_persistent);
  AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
}

void MaxPool2dWithIndicesOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  const auto dilation = inputs[4].toIntList().vec();
  bool ceil_mode = inputs[5].toBool();

  auto out_shape = compute_output_shape(
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

  HabanaOperator::SetPTOutputs({output_nhwc, output_idx_nhwc});
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

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[1], out_shape[2], out_shape[3]};
  TORCH_CHECK(grad_out.sizes().vec() == expected_output_size);
  TORCH_CHECK(input.sizes() == grad_input.sizes());
  TORCH_CHECK(grad_out.sizes() == indices.sizes());

  TORCH_CHECK(
      (indices.scalar_type() == c10::ScalarType::Byte) ||
      (indices.scalar_type() == c10::ScalarType::Short));

  HabanaOperator::SetPTOutputs({grad_input});
}

/**
 * @brief MaxPool2d.with_indices_hpu (Forward Pass) implementation for Habana
 * device
 * @param [In] Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] parameter that controls stride of elements in window. Default: 1
 * @param [In] when true use ceil instead of floor to compute output shape.
 * Default: false
 * @param [Out] Output Tensor. 4D, bf16/fp32
 * @param [Out] Output Indices. 1D, uint8
 */
std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  PT_KERNEL_BEGIN;
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  int device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "maxpool_2d_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  Tensor input_nhwc = input;
  std::vector<const at::Tensor*> pt_in = {&input};
  std::vector<at::Tensor*> pt_out = {&input_nhwc};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_in // = {0, 2, 3, 1};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, true);
  std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  auto maxpool_2d = [&] {
    std::vector<c10::IValue> stack = {IValue(input_nhwc),
                                      IValue(kernel_size),
                                      IValue(stride),
                                      IValue(padding),
                                      IValue(dilation),
                                      IValue(ceil_mode)};
    std::vector<at::Tensor> pt_inputs{input_nhwc};
    MaxPool2dWithIndicesOperator Op(device_id, scalar_type);
    size_t key = Op.GetRecipeKey(node_type, stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(stack);
      Op.Execute(key);

    } else {
      PT_KERNEL_DEBUG("key:", key);
      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      // Allocate synapse inputs
      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      // Build Params for the graph
      Op.AllocateAndAddSynapseNode(graph, stack, true);

      // compile and execute the graph
      Op.Compile(graph);
    }

    return Op.GetOutputs();
  };

  std::vector<at::Tensor> out = maxpool_2d();

  at::Tensor& output_idx_nhwc = out.at(1);
  at::Tensor& output_nhwc = out.at(0);

  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");
  at::Tensor output = output_nhwc;
  at::Tensor output_idx = output_idx_nhwc;
  pt_in = {&output_nhwc, &output_idx_nhwc};
  pt_out = {&output, &output_idx};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_out // = {0, 3, 1, 2};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, false);
  // Both the outputs require same layout, hence using the first output's
  // layout positions
  pt_new_pos = {&new_dim_pos_out, &new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  PT_KERNEL_END;
  return {output, output_idx};
}

/**
 * @brief MaxPool2d.with_indices_hpu.out (Backward Pass) implementation for
 * Habana device
 * @param [In/Out] Backward pass Output Tensor. 4D, bf16/fp32
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Indices Tensor. 1D, uint8
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] parameter that controls stride of elements in window. Default: 1
 * @param [In] when true use ceil instead of floor to compute output shape.
 * Default: false
 */
Tensor& max_pool2d_with_indices_backward_out_hpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  PT_KERNEL_BEGIN;
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  int device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "maxpool_2d_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor grad_input_nhwc = grad_input;
  Tensor grad_out_nhwc = grad_output;
  Tensor indices_nhwc = indices;
  std::vector<const at::Tensor*> pt_in{
      &input, &grad_input, &grad_output, &indices};
  std::vector<at::Tensor*> pt_out{
      &input_nhwc, &grad_input_nhwc, &grad_out_nhwc, &indices_nhwc};

  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_in // = {0, 2, 3, 1};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, true);
  std::vector<const IntArrayRef*> pt_new_pos{
      &new_dim_pos_in, &new_dim_pos_in, &new_dim_pos_in, &new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  auto maxpool_2d_bwd = [&] {
    std::vector<at::Tensor> pt_inputs{grad_out_nhwc, input_nhwc, indices_nhwc};
    std::vector<c10::IValue> stack = {IValue(grad_input_nhwc),
                                      IValue(grad_out_nhwc),
                                      IValue(input_nhwc),
                                      IValue(indices_nhwc),
                                      IValue(kernel_size),
                                      IValue(stride),
                                      IValue(padding),
                                      IValue(dilation),
                                      IValue(ceil_mode)};
    // Create the operator
    MaxPool2dWithIndicesBackwardOutOperator Op(device_id, scalar_type);
    size_t key = Op.GetRecipeKey(node_type, stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(stack);
      Op.Execute(key);

    } else {
      PT_KERNEL_DEBUG("key:", key);
      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      // Allocate synapse inputs
      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      // Build Params for the graph
      Op.AllocateAndAddSynapseNode(graph, stack, true);

      // compile and execute the graph
      Op.Compile(graph);
    }

    return Op.GetOutputs();
  };

  std::vector<at::Tensor> out = maxpool_2d_bwd();

  grad_input = out.at(0);
  pt_in = {&grad_input_nhwc};
  pt_out = {&grad_input};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_out // = {0, 3, 1, 2};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, false);
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  PT_KERNEL_END;
  return grad_input;
}

void MaxPool2dWithIndicesBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of input expected for MaxPool2dWithIndicesBackwardOperator");
  TORCH_CHECK(inputs[1].isTensor(), "Second input type expected to be tensor");

  at::Tensor input = inputs[1].toTensor();
  auto grad_input =
      at::empty_like(input, input.options(), input.suggest_memory_format());

  // Re-order the inpust for:
  // MaxPool2dWithIndicesBackwardOutOperator in the below order:
  // {grad_input, grad_out, input, indices, kernel_size, stride, padding,
  //  dialation, ceil_mode}
  inputs.insert(inputs.begin(), IValue(grad_input));
  auto& indices = inputs.back();
  inputs.pop_back();
  inputs.insert(inputs.begin() + 3, indices);

  MaxPool2dWithIndicesBackwardOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
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
  inputs.pop_back();
  inputs.insert(inputs.begin() + 3, indices);

  MaxPool2dWithIndicesBackwardOutOperator::SetPTOutputs(inputs);
}

/**
 * @brief MaxPool2d.with_indices_hpu (Backward Pass) implementation for Habana
 * device
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] parameter that controls stride of elements in window. Default: 1
 * @param [In] when true use ceil instead of floor to compute output shape.
 * Default: false
 * @param [In] Forward pass Indices Tensor. 1D, uint8
 * @param [Out] Backward pass Output Tensor. 4D, bf16/fp32
 */
Tensor max_pool2d_with_indices_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  PT_KERNEL_BEGIN;
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  int device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "maxpool_2d_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor grad_out_nhwc = grad_output;
  Tensor indices_nhwc = indices;
  std::vector<const at::Tensor*> pt_in{&input, &grad_output, &indices};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &grad_out_nhwc, &indices_nhwc};

  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_in // = {0, 2, 3, 1};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, true);
  std::vector<const IntArrayRef*> pt_new_pos{
      &new_dim_pos_in, &new_dim_pos_in, &new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  auto maxpool_2d_bwd = [&] {
    std::vector<at::Tensor> pt_inputs{grad_out_nhwc, input_nhwc, indices_nhwc};
    std::vector<c10::IValue> stack = {IValue(grad_out_nhwc),
                                      IValue(input_nhwc),
                                      IValue(kernel_size),
                                      IValue(stride),
                                      IValue(padding),
                                      IValue(dilation),
                                      IValue(ceil_mode),
                                      IValue(indices_nhwc)};
    // Create the operator
    MaxPool2dWithIndicesBackwardOperator Op(device_id, scalar_type);
    size_t key = Op.GetRecipeKey(node_type, stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(stack);
      Op.Execute(key);

    } else {
      PT_KERNEL_DEBUG("key:", key);
      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      // Allocate synapse inputs
      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      // Build Params for the graph
      Op.AllocateAndAddSynapseNode(graph, stack, true);

      // compile and execute the graph
      Op.Compile(graph);
    }

    return Op.GetOutputs();
  };

  std::vector<at::Tensor> out = maxpool_2d_bwd();

  Tensor grad_input;
  Tensor grad_input_nhwc = out.at(0);
  pt_in = {&grad_input_nhwc};
  pt_out = {&grad_input};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_out // = {0, 3, 1, 2};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, false);
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  PT_KERNEL_END;
  return grad_input;
}

void AvgPool2dOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  // Setup output tensors
  auto output_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options());

  AllocateSynapseOutputs(graph, {output_nhwc}, is_output_persistent);
  AddNodeToSynapseGraph(graph, &syn_pool_params, sizeof(syn_pool_params));
}

void AvgPool2dOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input = inputs[0].toTensor();
  const auto kernel_size = inputs[1].toIntList().vec();
  const auto stride = inputs[2].toIntList().vec();
  const auto padding = inputs[3].toIntList().vec();
  auto ceil_mode = inputs[4].toBool();
  auto divisor_override = inputs[6].toOptional<int64_t>();
  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  auto out_shape = compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, true);

  // Setup output tensors
  auto output_nhwc = at::empty(
      {out_shape[0], out_shape[1], out_shape[2], out_shape[3]},
      input.options());

  HabanaOperator::SetPTOutputs({output_nhwc});
}

/**
 * @brief AveragePool2d (Forward Pass) implementation for Habana device
 * @param [In] Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] when true use ceil instead of floor to compute output shape.
 * Default: false
 * @param [In] when true will include zero-padding in averaging. Default: true
 * @param [In] if specified, this value will be used as divisor. Default: None
 * @param [Out] Output Tensor. 4D, bf16/fp32
 */
Tensor avg_pool2d_hpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_KERNEL_BEGIN;

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  std::vector<const at::Tensor*> pt_in = {&input};
  std::vector<at::Tensor*> pt_out = {&input_nhwc};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_in // = {0, 2, 3, 1};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, true);
  std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  int device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "avg_pool_2d_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto avgpool_2d = [&] {
    std::vector<at::Tensor> pt_inputs{input_nhwc};
    // Build Params for the graph
    std::vector<c10::IValue> stack = {IValue(input_nhwc),
                                      IValue(kernel_size),
                                      IValue(stride),
                                      IValue(padding),
                                      IValue(ceil_mode),
                                      IValue(count_include_pad),
                                      IValue(divisor_override)};
    // Create the operator
    AvgPool2dOperator Op(device_id, scalar_type);
    size_t key = Op.GetRecipeKey(node_type, stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(stack);
      Op.Execute(key);
    } else {
      PT_KERNEL_DEBUG("key:", key);
      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      // Allocate synapse inputs
      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      // add node and allocate parms and output
      Op.AllocateAndAddSynapseNode(graph, stack, true);

      // compile and execute the graph
      Op.Compile(graph);
    }

    return Op.GetOutputs();
  };

  std::vector<at::Tensor> out = avgpool_2d();

  Tensor output = out.at(0);
  pt_in = {&out.at(0)};
  pt_out = {&output};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_out // = {0, 3, 1, 2};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, false);
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  PT_KERNEL_END;
  return output;
}

void AvgPool2dBackwardOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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

  auto out_shape = compute_output_shape(
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

  AllocateSynapseOutputs(graph, {grad_input_nhwc}, is_output_persistent);
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
  auto divisor_override = inputs[8].toOptional<int64_t>();

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());

  auto out_shape = compute_output_shape(
      input_nhwc, kernel_size, stride, padding, dilation, ceil_mode, true);
  std::vector<int64_t> expected_output_size{
      out_shape[0], out_shape[1], out_shape[2], out_shape[3]};

  TORCH_CHECK(
      input_nhwc.sizes() == grad_input_nhwc.sizes(),
      "Input and grad_input sizes don't match");
  TORCH_CHECK(grad_out_nhwc.sizes().vec() == expected_output_size);

  HabanaOperator::SetPTOutputs({grad_input_nhwc});
}

/**
 * @brief AveragePool2d.out (Backward Pass) implementation for Habana device
 * @param [In/Out] Backward pass Output Tensor. 4D, bf16/fp32
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] when true use ceil instead of floor to compute output shape.
 * Default: false
 * @param [In] when true will include zero-padding in averaging. Default: true
 * @param [In] if specified, this value will be used as divisor. Default: None
 */
Tensor& avg_pool2d_backward_out_hpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_KERNEL_BEGIN;

  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor grad_input_nhwc = grad_input;
  Tensor grad_out_nhwc = grad_output;
  std::vector<const at::Tensor*> pt_in{&input, &grad_input, &grad_output};
  std::vector<at::Tensor*> pt_out{
      &input_nhwc, &grad_input_nhwc, &grad_out_nhwc};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_in // = {0, 2, 3, 1};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, true);
  std::vector<const IntArrayRef*> pt_new_pos{
      &new_dim_pos_in, &new_dim_pos_in, &new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  int device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "avg_pool_2d_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto avgpool_bwd_out_2d = [&] {
    std::vector<at::Tensor> pt_inputs{grad_out_nhwc};
    // Build Params for the graph
    std::vector<c10::IValue> stack = {IValue(grad_input_nhwc),
                                      IValue(grad_out_nhwc),
                                      IValue(input_nhwc),
                                      IValue(kernel_size),
                                      IValue(stride),
                                      IValue(padding),
                                      IValue(ceil_mode),
                                      IValue(count_include_pad),
                                      IValue(divisor_override)};
    // Create the operator
    AvgPool2dBackwardOutOperator Op(device_id, scalar_type);
    size_t key = Op.GetRecipeKey(node_type, stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(stack);
      Op.Execute(key);

    } else {
      PT_KERNEL_DEBUG("key:", key);
      //
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      // Allocate synapse inputs
      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      Op.AllocateAndAddSynapseNode(graph, stack, true);

      // compile and execute the graph
      Op.Compile(graph);
    }

    return Op.GetOutputs();
  };

  std::vector<at::Tensor> out = avgpool_bwd_out_2d();

  pt_in = {&out.at(0)};
  pt_out = {&grad_input};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_out // = {0, 3, 1, 2};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, false);
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  PT_KERNEL_END;
  return grad_input;
}

void AvgPool2dBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(inputs[1].isTensor(), "Input1 type expected to be tensor");

  at::Tensor input_nhwc = inputs[1].toTensor();
  auto grad_input_nhwc = at::empty_like(input_nhwc, input_nhwc.options());

  inputs.insert(inputs.begin(), IValue(grad_input_nhwc));
  AvgPool2dBackwardOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void AvgPool2dBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  at::Tensor input_nhwc = inputs[1].toTensor();
  auto grad_input_nhwc = at::zeros_like(input_nhwc, input_nhwc.options());

  inputs.insert(inputs.begin(), IValue(grad_input_nhwc));
  AvgPool2dBackwardOutOperator::SetPTOutputs(inputs);
}

/**
 * @brief AveragePool2d (Backward Pass) implementation for Habana device
 * @param [In] Backward pass Input Tensor. 4D, bf16/fp32
 * @param [In] Forward pass Input Tensor. 4D, bf16/fp32
 * @param [In] size of the window. int64 or int64 tuple
 * @param [In] stride of the window. int64 or int64 tuple. Default: kernel_size
 * @param [In] zero padding on both sides. int64 or int64 tuple. Default: 0
 * @param [In] when true use ceil instead of floor to compute output shape.
 * Default: false
 * @param [In] when true will include zero-padding in averaging. Default: true
 * @param [In] if specified, this value will be used as divisor. Default: None
 * @param [Out] Backward pass Output Tensor. 4D, bf16/fp32
 */
Tensor avg_pool2d_backward_hpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_KERNEL_BEGIN;
  // Dilation set to 1, since for AvgPool Pytorch API does not give dilation
  // values
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());
  habana_helpers::check_pool_params(
      input, kernel_size, stride, padding, dilation, ceil_mode);

  // convert tensors to synapse memory format
  Tensor input_nhwc = input;
  Tensor grad_out_nhwc = grad_output;
  std::vector<const at::Tensor*> pt_in{&input, &grad_output};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &grad_out_nhwc};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_in // = {0, 2, 3, 1};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, true);
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in, &new_dim_pos_in};
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  int device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "avg_pool_2d_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto avgpool_bwd_2d = [&] {
    std::vector<at::Tensor> pt_inputs{grad_out_nhwc};
    // Build Params for the graph
    std::vector<c10::IValue> stack = {IValue(grad_out_nhwc),
                                      IValue(input_nhwc),
                                      IValue(kernel_size),
                                      IValue(stride),
                                      IValue(padding),
                                      IValue(ceil_mode),
                                      IValue(count_include_pad),
                                      IValue(divisor_override)};
    // Create the operator
    AvgPool2dBackwardOperator Op(device_id, scalar_type);
    size_t key = Op.GetRecipeKey(node_type, stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(stack);
      Op.Execute(key);

    } else {
      PT_KERNEL_DEBUG("key:", key);
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      // Allocate synapse inputs
      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      Op.AllocateAndAddSynapseNode(graph, stack, true);

      // compile and execute the graph
      Op.Compile(graph);
    }

    return Op.GetOutputs();
  };

  std::vector<at::Tensor> out = avgpool_bwd_2d();

  Tensor grad_input = out.at(0);
  pt_in = {&out.at(0)};
  pt_out = {&grad_input};
  // TBD: these layout requirements are properties of the operator, and should
  // be declared static class member variables rathe than per-object data.
  // Once this change is made, the layout would be retrieved from the operator
  // class.
  IntArrayRef new_dim_pos_out // = {0, 3, 1, 2};
      = HabanaOperator::getPermuteOrder(LayoutFormat::NHWC, false);
  pt_new_pos = {&new_dim_pos_out};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  PT_KERNEL_END;
  return grad_input;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::max_pool2d_with_indices",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MaxPool2dWithIndicesOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::max_pool2d_with_indices_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MaxPool2dWithIndicesBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::max_pool2d",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MaxPool2dOperator>(device_id, node_type);
            })
        .add(
            "aten::avg_pool2d",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<AvgPool2dOperator>(device_id, node_type);
            })
        .add(
            "aten::avg_pool2d_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<AvgPool2dBackwardOperator>(
                  device_id, node_type);
            });

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride = [], int[2] padding = 0, int[2] dilation = 1, bool ceil_mode = False) ->(Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(max_pool2d_with_indices_hpu),
                    &max_pool2d_with_indices_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(max_pool2d_with_indices_backward_hpu),
                    &max_pool2d_with_indices_backward_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(max_pool2d_with_indices_backward_out_hpu),
                    &max_pool2d_with_indices_backward_out_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(avg_pool2d_hpu),
                    &avg_pool2d_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(avg_pool2d_backward_hpu),
                    &avg_pool2d_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
