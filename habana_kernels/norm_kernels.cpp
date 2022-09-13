/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <ATen/ExpandUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>
#include <tuple>

#include "generated/ne.h"
#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lowering_util.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/unary_kernels.h"
#include "synapse_helpers/layout_utils.h"

using namespace torch;
using namespace habana;

bool is_5d_tensor(const std::vector<int64_t>& shape_in) {
  const uint64_t DIM5 = 5;
  return shape_in.size() == DIM5;
}

Tensor batch_norm_resize(
    const Tensor& input,
    uint num_out_dim,
    c10::MemoryFormat memory_format) {
  auto num_in_dim = input.dim();
  Tensor input_resize = at::alias(input.to(DeviceType::HPU));

  auto shape = DimVector(input_resize.sizes());
  auto strides = DimVector(input_resize.strides());
  switch (memory_format) {
    case c10::MemoryFormat::ChannelsLast: {
      if (num_out_dim > num_in_dim) {
        auto last = shape.back();
        shape.pop_back();
        // Create view_sizes initialized to part which has size=1 for upper dims
        auto view_sizes = std::vector<int64_t>(num_out_dim - num_in_dim, 1);
        // and append to shape
        shape.insert(shape.end(), view_sizes.begin(), view_sizes.end());
        shape.push_back(last);
        input_resize = input_resize.view(shape);
      } else if (num_out_dim < num_in_dim) {
        // Remove the additional x1 dimensions
        // TODO: The logic here won't work when size of any intermediate
        // (non-start,end) dimensions is 1
        std::vector<int64_t> new_shape;
        new_shape.push_back(shape[0]);
        for (uint cnt = 1; cnt < num_out_dim - 1; cnt++) {
          if (1 != shape.back())
            new_shape.push_back(shape[cnt]);
        }
        new_shape.push_back(shape[num_out_dim - 1]);
        input_resize = input_resize.view(new_shape);
      }
      break;
    }
    case c10::MemoryFormat::Contiguous: {
      if (num_out_dim > num_in_dim) {
        // Create view_sizes initialized to part which has size=1 for upper dims
        auto view_sizes = std::vector<int64_t>(num_out_dim - num_in_dim, 1);
        // and append to shape
        shape.insert(shape.end(), view_sizes.begin(), view_sizes.end());
        input_resize = input_resize.view(shape);
      } else if (num_out_dim < num_in_dim) {
        // Remove the additional x1 dimensions
        std::vector<int64_t> new_shape;
        for (uint cnt = 0; cnt < num_out_dim; cnt++) {
          new_shape.push_back(shape[cnt]);
        }
        input_resize = input_resize.view(new_shape);
      }
      break;
    }
    default:
      TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous");
  }
  return input_resize;
}

/**********************************************************************
*@brief Pushes the optional tensor to device if defined.Otherwise,
* create an empty device tensor
@param input - Optional 1D tensor
@param size - size of 1D tensor. used in case the tensor is not defined
@param device - Device param
@param output - 1D HPU tensor
**********************************************************************/
inline Tensor get_batch_norm_optional_tensors(
    const Tensor& input,
    uint size,
    Device device) {
  Tensor output;
  if (input.defined() == true) {
    return input.to(DeviceType::HPU);
  } else {
    output = at::empty(
        {size}, TensorOptions().dtype(c10::ScalarType::Float).device(device));
  }
  return output;
}

void BatchNormInfOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(in_stack[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(in_stack[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(in_stack[5].isBool(), "Input type expected to be bool");
  TORCH_CHECK(in_stack[6].isDouble(), "Input type expected to be double");
  TORCH_CHECK(in_stack[7].isDouble(), "Input type expected to be double");

  auto input = in_stack[0].toTensor();
  const auto momentum = in_stack[6].toDouble();
  const auto eps = in_stack[7].toDouble();

  auto output =
      habana_helpers::createPTTensor(input, output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  struct ns_BatchNormKernel::Params params;
  params.threshold.f = 0.0;
  params.momentum = static_cast<float>(momentum);
  params.epsilon = static_cast<float>(eps);
  p_context_->params_.emplace<ns_BatchNormKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

OutputShapeInfRetType BatchNormInfOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto input = inputs[0].toTensor();

  OutputShapeInfRetType out;
  out.AddOutputTensor(TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  return out;
}

std::vector<std::vector<int64_t>> LayerNormOperator::getOutputSizes(
    const at::Tensor& input,
    IntArrayRef normalized_shape) {
  auto output_sizes = input.sizes().vec();
  const auto input_shape = input.sizes();
  const int axis = input.dim() - normalized_shape.size();
  int64_t m =
      multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  std::vector<int64_t> shape_mean{1, 1, m, 1};
  return std::vector<std::vector<int64_t>>{
      output_sizes, shape_mean, shape_mean};
}

std::tuple<Tensor, Tensor, Tensor> LayerNormOperator::AllocatePTOutputs(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& bias,
    const Tensor& weight,
    UNUSED int64_t m,
    std::array<bool, 3> is_persistent) {
  auto sizes = LayerNormOperator::getOutputSizes(input, normalized_shape);
  auto output = habana_helpers::createPTTensor(
      input,
      sizes[0],
      input.options(),
      input.suggest_memory_format(),
      is_persistent[0]);
  auto istd = habana_helpers::createPTTensor(
      bias,
      sizes[1],
      bias.options(),
      bias.suggest_memory_format(),
      is_persistent[1]);
  auto mean = habana_helpers::createPTTensor(
      weight,
      sizes[2],
      weight.options(),
      weight.suggest_memory_format(),
      is_persistent[2]);

  return std::make_tuple(std::move(output), std::move(mean), std::move(istd));
}

void LayerNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "LayerNormOperator::AllocateAndAddSynapseNode expected 5 args but got ",
      inputs.size())

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isIntList(), "Input type expected to be int list");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isDouble(), "Input type expected to be double");
  const auto input = inputs[0].toTensor();
  const auto normalized_shape = inputs[1].toIntList().vec();
  const auto weight = inputs[2].toTensor();
  if (is_tpc_affine_path(input, normalized_shape, weight)) {
    // this is the case where we can use the TPC layer_norm_fwd_affine path
    // which can efficiently divide load across TPCs
    AllocateAndAddSynapseNodeTPCAffinePath(graph, inputs, output_metadata);
  } else {
    AllocateAndAddSynapseNodeReshapePath(graph, inputs, output_metadata);
  }
}

/*
Nodes in LayerNorm graph
input->[reshape_in]--------->|----------------|
bias->[reshape_bias]-------->| LayerNorm Node |-->[reshape_ln_out]->final outs
weight->[reshape_weight]---->|----------------|
                                        ^
                                        |
{output}->[reshape_output]->{reshaped_output,mean,istd}
*/
void LayerNormOperator::AllocateAndAddSynapseNodeReshapePath(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  OutputMetaDataVector output_metadata_all_outputs = output_metadata;
  if (output_metadata.size() == 1) {
    output_metadata_all_outputs =
        OutputMetaDataVector(3, output_metadata.at(0));
  }

  const auto input = inputs[0].toTensor();
  const auto normalized_shape = inputs[1].toIntList().vec();
  const auto weight = inputs[2].toTensor();
  const auto bias = inputs[3].toTensor();
  const auto eps = inputs[4].toDouble();

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  const int normalized_ndim = normalized_shape.size();
  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  int64_t m =
      multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  int64_t n =
      multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  // Add Reshape node for input to graph for input.view({m,n})
  auto reshape_op_input = make_operator<ReshapeOperator>(
      input.device().index(), input.scalar_type());
  reshape_op_input->SetSynapseInput(p_context_->syn_inputs_[0]);
  int64_t modified_input_sizes[] = {1, 1, m, n};
  c10::IntArrayRef modified_input_shape(modified_input_sizes, 4);
  torch::jit::Stack stack = {
      c10::IValue(input), c10::IValue(modified_input_shape)};
  reshape_op_input->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  auto input_reshaped = reshape_op_input->GetOutputs()[0];
  synapse_helpers::tensor& syn_in_ln = reshape_op_input->GetSynOutputs()[0];
  // Add Reshape node for bias to graph for bias.view(-1)
  auto reshape_op_bias =
      make_operator<ReshapeOperator>(bias.device().index(), bias.scalar_type());
  reshape_op_bias->SetSynapseInput(p_context_->syn_inputs_[2]);
  int64_t sizes[1];
  sizes[0] = bias.numel();
  c10::IntArrayRef modified_bias_shape(sizes, 1);
  stack = {c10::IValue(bias), c10::IValue(modified_bias_shape)};
  reshape_op_bias->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  auto bias_reshaped = reshape_op_bias->GetOutputs()[0];
  synapse_helpers::tensor& syn_bias_ln = reshape_op_bias->GetSynOutputs()[0];

  // Add Reshape node for weight to graph for weight.view(-1)
  auto reshape_op_wt = make_operator<ReshapeOperator>(
      weight.device().index(), weight.scalar_type());
  reshape_op_wt->SetSynapseInput(p_context_->syn_inputs_[1]);
  sizes[0] = weight.numel();
  c10::IntArrayRef modified_weight_shape(sizes, 1);
  stack = {c10::IValue(weight), c10::IValue(modified_weight_shape)};
  reshape_op_wt->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  auto wt_reshaped = reshape_op_wt->GetOutputs()[0];
  synapse_helpers::tensor& syn_wt_ln = reshape_op_wt->GetSynOutputs()[0];

  auto outputs = AllocatePTOutputs(
      input,
      normalized_shape,
      bias_reshaped,
      wt_reshaped,
      m,
      {false,
       output_metadata_all_outputs.at(1).persistent,
       output_metadata_all_outputs.at(2).persistent});
  auto output = std::get<0>(outputs);
  auto mean = std::get<1>(outputs);
  auto istd = std::get<2>(outputs);

  std::vector<synTensor> syn_inputs{syn_in_ln.get()};
  syn_inputs.push_back(syn_bias_ln.get());
  syn_inputs.push_back(syn_wt_ln.get());
  // output syn tensor is non-persistent since it will be reshaped to input
  // sizes which will be marked as persistent
  AllocateSynapseOutput(
      graph,
      habana_helpers::createPTTensor(input_reshaped, false),
      OutputMetaData());
  AllocateSynapseOutput(graph, mean, output_metadata_all_outputs.at(1));
  AllocateSynapseOutput(graph, istd, output_metadata_all_outputs.at(2));
  synapse_helpers::tensor& syn_out_ln_out = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{syn_out_ln_out.get()};
  synapse_helpers::tensor& syn_out_ln_mean = p_context_->syn_outputs_[1];
  syn_outputs.push_back(syn_out_ln_mean.get());
  synapse_helpers::tensor& syn_out_ln_istd = p_context_->syn_outputs_[2];
  syn_outputs.push_back(syn_out_ln_istd.get());

  struct ns_LayerNormKernel::Params params;
  params.eps = static_cast<float>(eps);
  params.epsValid = true;

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      guid_,
      nullptr,
      nullptr,
      nullptr,
      deterministic);
  // Add Reshape node for output tensor to graph -
  // output.view(input.sizes().vec())
  auto reshape_op_out = make_operator<ReshapeOperator>(
      input.device().index(), input.scalar_type());
  reshape_op_out->SetSynapseInput(p_context_->syn_outputs_[0]);
  stack = {c10::IValue(input_reshaped), c10::IValue(input.sizes().vec())};
  reshape_op_out->AllocateAndAddSynapseNode(
      graph, stack, {output_metadata_all_outputs.at(0)});
  synapse_helpers::tensor& syn_reshape_out = reshape_op_out->GetSynOutputs()[0];
  p_context_->syn_outputs_[0] = std::move(syn_reshape_out);
  p_context_->pt_outputs_[0] = reshape_op_out->GetOutputs()[0];
}

void LayerNormOperator::AllocateAndAddSynapseNodeTPCAffinePath(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  OutputMetaDataVector output_metadata_all_outputs = output_metadata;
  if (output_metadata.size() == 1) {
    output_metadata_all_outputs =
        OutputMetaDataVector(3, output_metadata.at(0));
  }

  const auto input = inputs[0].toTensor();
  auto normalized_shape = inputs[1].toIntList().vec();
  const auto weight = inputs[2].toTensor();
  const auto bias = inputs[3].toTensor();
  const auto eps = inputs[4].toDouble();

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  normalized_shape.erase(
      normalized_shape
          .begin()); // Lazy frontend would have inserted an additional element
                     // to indicate elementwise_affine=false. Remove this
  const int normalized_ndim = normalized_shape.size();
  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }

  const int axis = input_ndim - normalized_ndim;
  int64_t m =
      multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);

  torch::jit::Stack stack;
  synapse_helpers::tensor& syn_in_ln = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& syn_wt_ln = p_context_->syn_inputs_[1];
  synapse_helpers::tensor& syn_bias_ln = p_context_->syn_inputs_[2];

  auto outputs = AllocatePTOutputs(
      input,
      normalized_shape,
      bias,
      weight,
      m,
      {output_metadata_all_outputs.at(0).persistent, // false
       output_metadata_all_outputs.at(1).persistent,
       output_metadata_all_outputs.at(2).persistent});
  auto output = std::get<0>(outputs);
  auto mean = std::get<1>(outputs);
  auto istd = std::get<2>(outputs);
  std::vector<synTensor> syn_inputs{syn_in_ln.get()};
  syn_inputs.push_back(syn_bias_ln.get());
  syn_inputs.push_back(syn_wt_ln.get());
  // output syn tensor is non-persistent since it will be reshaped to input
  // sizes which will be marked as persistent
  AllocateSynapseOutput(graph, output, output_metadata_all_outputs.at(0));
  AllocateSynapseOutput(graph, mean, output_metadata_all_outputs.at(1));
  AllocateSynapseOutput(graph, istd, output_metadata_all_outputs.at(2));
  synapse_helpers::tensor& syn_out_ln_out = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{syn_out_ln_out.get()};
  synapse_helpers::tensor& syn_out_ln_mean = p_context_->syn_outputs_[1];
  syn_outputs.push_back(syn_out_ln_mean.get());
  synapse_helpers::tensor& syn_out_ln_istd = p_context_->syn_outputs_[2];
  syn_outputs.push_back(syn_out_ln_istd.get());

  struct ns_LayerNormKernel::ParamsNorm params;
  params.eps = static_cast<float>(eps);
  params.epsValid = true;
  params.NormAxisBmp =
      (1 << normalized_shape.size()) - 1; // normalize across CWH
  params.ParamAxisBmp = 1;
  auto input_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      kernel_meta_data_.synapse_input_layout);
  auto output_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      kernel_meta_data_.synapse_output_layout);
  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      guid_,
      nullptr,
      input_layouts.data(),
      output_layouts.data(),
      deterministic);
}

std::tuple<Tensor, Tensor, Tensor> LayerNormBackwardOperator::AllocatePTOutputs(
    const Tensor& input,
    const Tensor& weight,
    bool is_persistent) {
  auto sizes = LayerNormBackwardOperator::getOutputSizes(input, weight);
  auto output = habana_helpers::createPTTensor(
      input,
      sizes[0],
      input.options(),
      input.suggest_memory_format(),
      is_persistent);
  auto beta = habana_helpers::createPTTensor(
      weight,
      sizes[1],
      weight.options(),
      weight.suggest_memory_format(),
      is_persistent);
  auto gamma = habana_helpers::createPTTensor(
      weight,
      sizes[2],
      weight.options(),
      weight.suggest_memory_format(),
      is_persistent);

  return std::make_tuple(std::move(output), std::move(beta), std::move(gamma));
}

void LayerNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 8,
      "LayerNormBackwardOperator::AllocateAndAddSynapseNode expected 8 args but got ",
      inputs.size());
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isIntList(), "Input type expected to be int list");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[6].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[7].isBoolList(), "Input type expected to be bool array");

  OutputMetaDataVector output_metadata_all_outputs = output_metadata;
  if (output_metadata.size() == 1) {
    output_metadata_all_outputs =
        OutputMetaDataVector(3, output_metadata.at(0));
  }

  const auto dY = inputs[0].toTensor();
  const auto X = inputs[1].toTensor();
  const auto normalized_shape = inputs[2].toIntList().vec();
  const auto mean = inputs[3].toTensor();
  const auto rstd = inputs[4].toTensor();
  const auto gamma = inputs[5].toTensor();

  const auto grad_input_mask = inputs[7].toBoolList();

  const auto input_shape = X.sizes();
  const auto input_ndim = X.dim();
  const int normalized_ndim = normalized_shape.size();
  const int axis = input_ndim - normalized_ndim;
  int64_t m =
      multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
  int64_t n =
      multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

  // Add Reshape node for input to graph for input.view({m,n})
  auto reshape_op_x =
      make_operator<ReshapeOperator>(X.device().index(), X.scalar_type());
  reshape_op_x->SetSynapseInput(p_context_->syn_inputs_[1]);
  // TPC Kernel needs 4D inputs
  std::array<int64_t, 4> modified_x_sizes = {1, 1, m, n};
  c10::IntArrayRef modified_x_shape(
      modified_x_sizes.data(), modified_x_sizes.size());
  torch::jit::Stack stack = {c10::IValue(X), c10::IValue(modified_x_shape)};
  reshape_op_x->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  synapse_helpers::tensor& syn_x = reshape_op_x->GetSynOutputs()[0];
  stack.clear();

  // Add Reshape node for input to graph for grad_in.view({m,n})
  auto reshape_op_dy =
      make_operator<ReshapeOperator>(dY.device().index(), dY.scalar_type());
  reshape_op_dy->SetSynapseInput(p_context_->syn_inputs_[0]);
  // TPC Kernel needs 4D inputs
  std::array<int64_t, 4> modified_dy_sizes = {1, 1, m, n};
  c10::IntArrayRef modified_dy_shape(
      modified_dy_sizes.data(), modified_dy_sizes.size());
  stack = {c10::IValue(dY), c10::IValue(modified_dy_shape)};
  reshape_op_dy->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  auto dy_reshaped = reshape_op_dy->GetOutputs()[0];
  synapse_helpers::tensor& syn_dy = reshape_op_dy->GetSynOutputs()[0];
  stack.clear();

  // Add Reshape node for weight to graph for weight.view(-1)
  auto reshape_op_gamma = make_operator<ReshapeOperator>(
      gamma.device().index(), gamma.scalar_type());
  reshape_op_gamma->SetSynapseInput(p_context_->syn_inputs_[4]);
  std::array<int64_t, 1> sizes;
  sizes[0] = gamma.numel();
  c10::IntArrayRef modified_gamma_shape(sizes.data(), sizes.size());
  stack = {c10::IValue(gamma), c10::IValue(modified_gamma_shape)};
  reshape_op_gamma->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  auto gamma_reshaped = reshape_op_gamma->GetOutputs()[0];
  synapse_helpers::tensor& syn_gamma = reshape_op_gamma->GetSynOutputs()[0];
  stack.clear();

  // Add reshape node for mean
  auto reshape_op_mean =
      make_operator<ReshapeOperator>(mean.device().index(), mean.scalar_type());
  reshape_op_mean->SetSynapseInput(p_context_->syn_inputs_[2]);
  // TPC Kernel needs 4D inputs - Reshape if needed.
  TORCH_CHECK(
      mean.sizes().size() <= 4,
      "Input mean for LayerNormBackward is over 4 dims - unsupported!");

  std::array<int64_t, 4> modified_mean_sizes = {1, 1, m, 1};
  c10::IntArrayRef modified_mean_shape(
      modified_mean_sizes.data(), modified_mean_sizes.size());
  stack = {c10::IValue(mean), c10::IValue(modified_mean_shape)};
  reshape_op_mean->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  synapse_helpers::tensor& syn_mean = reshape_op_mean->GetSynOutputs()[0];
  stack.clear();

  // Add reshape node for rstd
  auto reshape_op_rstd =
      make_operator<ReshapeOperator>(rstd.device().index(), rstd.scalar_type());
  reshape_op_rstd->SetSynapseInput(p_context_->syn_inputs_[3]);
  // TPC Kernel needs 4D inputs - Reshape if needed.
  TORCH_CHECK(
      rstd.sizes().size() <= 4,
      "Input rstd for LayerNormBackward is over 4 dims - unsupported!");

  std::array<int64_t, 4> modified_rstd_sizes = {1, 1, m, 1};
  c10::IntArrayRef modified_rstd_shape(
      modified_rstd_sizes.data(), modified_rstd_sizes.size());
  stack = {c10::IValue(rstd), c10::IValue(modified_rstd_shape)};
  reshape_op_rstd->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  synapse_helpers::tensor& syn_rstd = reshape_op_rstd->GetSynOutputs()[0];
  stack.clear();
  // Add layer_norm_bwd node to graph
  std::vector<synTensor> syn_inputs{syn_x.get(), syn_dy.get()};
  syn_inputs.push_back(syn_mean.get());
  syn_inputs.push_back(syn_rstd.get());
  syn_inputs.push_back(syn_gamma.get());
  // output syn tensors are non-persistent since these will be reshaped to
  // input sizes which will be marked as persistent
  auto outputs = AllocatePTOutputs(dY, gamma, false);
  auto output0 = std::get<0>(outputs);
  auto output1 = std::get<1>(outputs);
  auto output2 = std::get<2>(outputs);
  AllocateSynapseOutput(
      graph,
      habana_helpers::createPTTensor(dy_reshaped, false),
      OutputMetaData(),
      false);
  AllocateSynapseOutput(
      graph,
      habana_helpers::createPTTensor(gamma_reshaped, false),
      OutputMetaData(),
      false);
  AllocateSynapseOutput(
      graph,
      habana_helpers::createPTTensor(gamma_reshaped, false),
      OutputMetaData(),
      false);
  synapse_helpers::tensor& syn_grad_out = p_context_->syn_outputs_[0];
  synapse_helpers::tensor& syn_grad_beta = p_context_->syn_outputs_[1];
  synapse_helpers::tensor& syn_grad_gamma = p_context_->syn_outputs_[2];
  std::vector<synTensor> syn_outputs{
      syn_grad_out.get(), syn_grad_beta.get(), syn_grad_gamma.get()};
  struct ns_LayerNormKernel::Params params;
  params.epsValid = false;
  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      guid_,
      nullptr,
      nullptr,
      nullptr,
      deterministic);

  // Add Reshape nodes for output tensors to graph
  auto reshape_op_grad_in =
      make_operator<ReshapeOperator>(dY.device().index(), dY.scalar_type());
  reshape_op_grad_in->SetSynapseInput(p_context_->syn_outputs_[0]);
  stack = {c10::IValue(output0), c10::IValue(output0.sizes().vec())};
  reshape_op_grad_in->AllocateAndAddSynapseNode(
      graph, stack, SelectVectorIndices(output_metadata_all_outputs, {0}));
  synapse_helpers::tensor& syn_reshape_grad_in =
      reshape_op_grad_in->GetSynOutputs()[0];

  auto reshape_op_grad_gamma = make_operator<ReshapeOperator>(
      gamma.device().index(), gamma.scalar_type());
  reshape_op_grad_gamma->SetSynapseInput(p_context_->syn_outputs_[2]);
  stack = {c10::IValue(output2), c10::IValue(output2.sizes().vec())};
  reshape_op_grad_gamma->AllocateAndAddSynapseNode(
      graph, stack, SelectVectorIndices(output_metadata_all_outputs, {1}));
  synapse_helpers::tensor& syn_reshape_grad_gamma =
      reshape_op_grad_gamma->GetSynOutputs()[0];

  auto reshape_op_grad_beta = make_operator<ReshapeOperator>(
      gamma.device().index(), gamma.scalar_type());
  reshape_op_grad_beta->SetSynapseInput(p_context_->syn_outputs_[1]);
  stack = {c10::IValue(output1), c10::IValue(output1.sizes().vec())};
  reshape_op_grad_beta->AllocateAndAddSynapseNode(
      graph, stack, SelectVectorIndices(output_metadata_all_outputs, {2}));
  synapse_helpers::tensor& syn_reshape_grad_beta =
      reshape_op_grad_beta->GetSynOutputs()[0];

  // beta & gamma are produced by TPC kernel in order opposite to
  // that required by PyTorch. Take care of swapping into correct
  // order here.
  p_context_->syn_outputs_[0] = std::move(syn_reshape_grad_in);
  p_context_->pt_outputs_[0] = reshape_op_grad_in->GetOutputs()[0];
  p_context_->syn_outputs_[1] = std::move(syn_reshape_grad_gamma);
  p_context_->pt_outputs_[1] = reshape_op_grad_gamma->GetOutputs()[0];
  p_context_->syn_outputs_[2] = std::move(syn_reshape_grad_beta);
  p_context_->pt_outputs_[2] = reshape_op_grad_beta->GetOutputs()[0];
}

std::vector<std::vector<int64_t>> LayerNormBackwardOperator::getOutputSizes(
    const at::Tensor& input,
    const at::Tensor& gamma) {
  std::vector<int64_t> gamma_size =
      gamma.defined() ? gamma.sizes().vec() : input.sizes().vec();
  return std::vector<std::vector<int64_t>>{
      input.sizes().vec(), gamma_size, gamma_size};
}

std::vector<int64_t> NormOperator::compute_output_shape(
    const Tensor& self,
    at::IntArrayRef dim,
    bool keepdim) {
  if (dim.size() == 0)
    return {};
  auto sizes = self.sizes().vec();
  std::vector<int64_t> wrapped_dims;
  for (unsigned i = 0; i < dim.size(); i++)
    wrapped_dims.emplace_back(at::maybe_wrap_dim(dim[i], self.dim()));
  unsigned removed_count = 0;
  for (unsigned i = 0; i < wrapped_dims.size(); i++) {
    if (keepdim) {
      sizes[wrapped_dims[i]] = 1;
    } else {
      sizes.erase(sizes.cbegin() + wrapped_dims[i] - removed_count);
      removed_count++;
    }
  }
  return sizes;
}

void NormOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() >= 2 && inputs.size() <= 4,
      "Incorrect size of inputs expected for Norm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for Norm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for Norm Operator");

  auto self = inputs[0].toTensor();
  auto shape = NormOperator::compute_output_shape(self, {}, 0);
  auto output = at::empty(shape, self.options(), c10::nullopt);
  HabanaOperator::SetPTOutput(output);
}

struct NotEqualScalar : NE {
  NotEqualScalar(int device_id, c10::ScalarType scalar_type)
      : NE(device_id, "None_", scalar_type, {0}, {}, {1}, false) {
    EnableTypePromotion();
  }
};

void NormOperator::AddL0NormNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> dims;
  bool keepdim;
  if (inputs.size() == 2) { // reduce along all dims
    for (unsigned i = 0; i < inputs[0].toTensor().sizes().size(); i++)
      dims.emplace_back(i);
    keepdim = false;
  } else { // reduce along given dims
    dims = inputs[2].toIntList().vec();
    keepdim = inputs[3].toBool();
  }
  auto device_id = self.device().index();
  auto scalar_type = self.scalar_type();
  torch::jit::Stack stack;
  auto ne_op = make_operator<NotEqualScalar>(device_id, scalar_type);
  ne_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(0.0));
  ne_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();
  std::string node_type =
      "cast_i8_to_" + habana_helpers::name_suffix_from_type(scalar_type);
  auto cast1 = make_operator<CastOperator>(device_id, node_type);
  cast1->SetSynapseInput(ne_op->GetSynOutputs()[0]);
  stack.emplace_back(IValue(ne_op->GetOutputs()[0]));
  stack.emplace_back(IValue(scalar_type));
  cast1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();
  // Reduction operation - Create the operator
  auto sum_dim_op =
      make_operator<SumDimOperator>(this->p_context_->device_id_, scalar_type);
  sum_dim_op->SetSynapseInput(cast1->GetSynOutputs()[0]);
  stack.emplace_back(IValue(cast1->GetOutputs()[0]));
  stack.emplace_back(IValue(dims));
  stack.emplace_back(IValue(keepdim));
  stack.emplace_back(IValue(scalar_type));
  sum_dim_op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  stack.clear();
  p_context_->syn_outputs_.emplace_back(
      std::move(sum_dim_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(sum_dim_op->GetOutputs()[0]);
}

void NormOperator::AddLInfNormNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaData& output_metadata) {
  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();
  std::vector<int64_t> dims;
  bool keepdim;
  if (inputs.size() == 2) { // reduce along all dims
    for (unsigned i = 0; i < inputs[0].toTensor().sizes().size(); i++)
      dims.emplace_back(i);
    keepdim = false;
  } else { // reduce along given dims
    dims = inputs[2].toIntList().vec();
    keepdim = inputs[3].toBool();
  }
  auto device_id = self.device().index();
  auto scalar_type = self.scalar_type();
  torch::jit::Stack stack;
  auto abs_op = make_operator<AbsOperator>(device_id, scalar_type);
  abs_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  stack.emplace_back(IValue(self));
  abs_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();
  // Reduction operation - Create the operator
  std::shared_ptr<ReduceOperator> reduce_op;
  if (p.toFloat() == LoweringUtil::FP_INFINITY) {
    reduce_op = make_operator<ReduceMultiOutputOperator>(
        this->p_context_->device_id_, scalar_type, "max");
  } else if (p.toFloat() == LoweringUtil::FP_NEG_INFINITY) {
    reduce_op = make_operator<ReduceMultiOutputOperator>(
        this->p_context_->device_id_, scalar_type, "min");
  } else {
    HABANA_ASSERT(0, "Call to AddLInfNormNode with invalid p value");
  }
  reduce_op->SetSynapseInput(abs_op->GetSynOutputs()[0]);
  stack.emplace_back(IValue(abs_op->GetOutputs()[0]));
  stack.emplace_back(IValue(dims));
  stack.emplace_back(IValue(keepdim));
  stack.emplace_back(IValue(scalar_type));
  reduce_op->AllocateAndAddSynapseNode(graph, stack, {output_metadata});
  stack.clear();
  p_context_->syn_outputs_.emplace_back(
      std::move(reduce_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(reduce_op->GetOutputs()[0]);
}

void NormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() >= 2 && inputs.size() <= 5,
      "Incorrect size of inputs expected for Norm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for Norm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for Norm Operator");
  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();
  if (p.toFloat() == 0.0) {
    // L0 Norm
    AddL0NormNode(graph, inputs, output_metadata.at(0));
    return;
  } else if (
      p.toFloat() == LoweringUtil::FP_INFINITY ||
      p.toFloat() == LoweringUtil::FP_NEG_INFINITY) {
    // LInf and LNegInf Norms
    AddLInfNormNode(graph, inputs, output_metadata.at(0));
    return;
  }
  if ((p.toFloat() == 2.0) && (inputs.size() < 3)) {
    if (self.dim() <= 1 || self.sizes()[0] == 1) {
      auto device_id = self.device().index();
      auto scalar_type = self.scalar_type();
      // x^2 implemented as x*x. Identity node used to create aliased tensor
      // since GC/TPC does not like giving same tensor as both inputs to a
      // binary op
      auto identityOp = make_operator<IdentityOperator>(
          this->p_context_->device_id_, scalar_type);
      identityOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      torch::jit::Stack stack = {IValue(self)};
      identityOp->AllocateAndAddSynapseNode(
          graph, stack, OutputMetaDataVector(1));
      stack.clear();

      auto mulOp = make_operator<habana::MulOperator>(
          this->p_context_->device_id_, scalar_type);
      mulOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      mulOp->SetSynapseInput(identityOp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(self));
      stack.emplace_back(IValue(identityOp->GetOutputs()[0]));
      mulOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
      // add node to compute reduce_sum
      auto sum_lp = make_operator<SumOperator>(device_id, scalar_type);
      sum_lp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
      stack.emplace_back(IValue(scalar_type));
      sum_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      auto sqrt_op = make_operator<SqrtOperator>(device_id, scalar_type);
      sqrt_op->SetSynapseInput(sum_lp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(sum_lp->GetOutputs()[0]));
      sqrt_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
      stack.clear();
      // synapse_helpers::tensor& sum_syn_tensor = sum_lp.GetSynOutputs()[0];
      p_context_->syn_outputs_.emplace_back(
          std::move(sqrt_op->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(sqrt_op->GetOutputs()[0]);
    } else {
      at::ScalarType scalar_type = self.scalar_type();
      std::vector<c10::IValue> stack{};
      // LpNorm Operator
      // Create the operator
      auto LpNormFrobeniusOp = make_operator<LpNormFrobeniusOperator>(
          this->p_context_->device_id_, scalar_type);
      LpNormFrobeniusOp->SetSynapseInput(p_context_->syn_inputs_[0]);

      // Build Params for the graph
      stack.emplace_back(IValue(self));
      LpNormFrobeniusOp->AllocateAndAddSynapseNode(
          graph, stack, output_metadata);

      stack.clear();
      p_context_->syn_outputs_.emplace_back(
          std::move(LpNormFrobeniusOp->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(LpNormFrobeniusOp->GetOutputs()[0]);
    }
  } else if (inputs.size() < 3) { // no dims given - reduce all dims
    // ReShape Operator
    at::ScalarType scalar_type = self.scalar_type();
    auto shape = {self.numel()};
    auto ReShapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, scalar_type);
    ReShapeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::IntArrayRef(shape))};
    ReShapeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

    auto output_reshape = ReShapeOp->GetOutputs()[0];
    stack.clear();
    // LpNorm Operator
    // Create the operator
    auto LpNormOp = make_operator<LpNormOperator>(
        this->p_context_->device_id_, scalar_type);
    LpNormOp->SetSynapseInput(ReShapeOp->GetSynOutputs()[0]);
    // Build Params for the graph
    stack.emplace_back(IValue(output_reshape));
    stack.emplace_back(IValue(p));
    stack.emplace_back(IValue(0)); // dim
    LpNormOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_.emplace_back(
        std::move(LpNormOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(LpNormOp->GetOutputs()[0]));
  } else { // reduce along specified dim
    at::ScalarType scalar_type = self.scalar_type();
    std::vector<int64_t> dims = inputs[2].toIntList().vec();
    bool keepdim = inputs[3].toBool();
    std::vector<c10::IValue> stack;
    if (dims.size() == 1) {
      auto LpNormOp = make_operator<LpNormOperator>(
          this->p_context_->device_id_, scalar_type);
      LpNormOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      stack = {IValue(self), IValue(p), IValue(dims[0])};
      LpNormOp->AllocateAndAddSynapseNode(
          graph, stack, keepdim ? output_metadata : OutputMetaDataVector(1));

      if (!keepdim) {
        auto new_sizes = LpNormOp->GetOutputs()[0].sizes().vec();
        new_sizes.erase(new_sizes.cbegin() + dims[0]);
        auto reshape_op = make_operator<ReshapeOperator>(
            this->p_context_->device_id_, scalar_type);
        reshape_op->SetSynapseInput(LpNormOp->GetSynOutputs()[0]);
        stack = {IValue(LpNormOp->GetOutputs()[0]), IValue(new_sizes)};
        reshape_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
        p_context_->syn_outputs_.emplace_back(
            std::move(reshape_op->GetSynOutputs()[0]));
        p_context_->pt_outputs_.emplace_back(
            std::move(reshape_op->GetOutputs()[0]));
      } else {
        p_context_->syn_outputs_.emplace_back(
            std::move(LpNormOp->GetSynOutputs()[0]));
        p_context_->pt_outputs_.emplace_back(
            std::move(LpNormOp->GetOutputs()[0]));
      }
    } else {
      HABANA_ASSERT(
          0, "NormOperator doesn't support more than single dim reduction");
    }

    return;
  }
}

void LpNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for LpNorm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for LpNorm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for LpNorm Operator");
  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();
  auto dim = inputs[2].toInt();

  TORCH_CHECK(p.toFloat() > 0.0, "norm with p > 0.0 is only supported");
  auto lpnorm_output = habana_helpers::createPTTensor(self, false);
  auto retain = habana_helpers::createPTTensor(self, false);

  ns_LpNormKernel::Params params{};
  params.p = p.to<float>();
  params.dim = self.dim() - dim - 1;
  params.eps = 1e-5; // arbitrarily small value
  std::vector<at::Tensor> outputs{lpnorm_output, retain};
  AllocateSynapseOutputs(graph, outputs, OutputMetaDataVector(2));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  std::vector<c10::IValue> stack;
  // Reciprocal Operator is used since what we require is reciprocal of retain
  auto reciprocalOp = make_operator<ReciprocalOperator>(
      this->p_context_->device_id_, self.scalar_type());
  reciprocalOp->SetSynapseInput(p_context_->syn_outputs_[1]);
  // Build Params for the graph
  stack.emplace_back(IValue(retain));
  reciprocalOp->AllocateAndAddSynapseNode(
      graph, stack, OutputMetaDataVector(1));
  stack.clear();
  // take just the first element of Reciprocal since all values would be
  // repeated
  auto slice_op = make_operator<SliceOperator>(
      this->p_context_->device_id_, self.scalar_type());
  slice_op->SetSynapseInput(reciprocalOp->GetSynOutputs()[0]);
  stack.emplace_back(IValue(reciprocalOp->GetOutputs()[0]));
  int start = 0;
  int end = 1;
  int step = 1;
  stack.emplace_back(IValue(dim));
  stack.emplace_back(IValue(start));
  stack.emplace_back(IValue(end));
  stack.emplace_back(IValue(step));
  slice_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  p_context_->syn_outputs_.erase(p_context_->syn_outputs_.begin());
  p_context_->syn_outputs_.insert(
      p_context_->syn_outputs_.begin(),
      std::move(slice_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.erase(p_context_->pt_outputs_.begin());
  p_context_->pt_outputs_.insert(
      p_context_->pt_outputs_.begin(), std::move(slice_op->GetOutputs()[0]));
}

void LpNormFrobeniusOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for LpNorm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for LpNorm Operator");

  auto self = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(
      self,
      {1},
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for LP Norm (Frobenius norm) kernel
          output = torch.norm(self, p=2)
 * @param [in] self - input tensor, 1-4D, FP32/BF16
 * @param [in] output - output tensor, 1-4D, FP32/BF16
 * @param [in] p - optional input, default = 2
 ************************************************************************/
Tensor norm_scalar_hpu(const Tensor& self, Scalar p) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "lpnorm_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  NormOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(p)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    Op.Execute(key, pt_inputs, stack);
  } else {
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  SET_SIZE_STRIDE_0D(out.at(0));

  PT_KERNEL_END;
  return out.at(0);
}

std::shared_ptr<SliceOperator> FusedNormOperator::compute_clip_coeff(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto gradients = inputs[0].toTensorList();
  auto max_grad_norm = inputs[1].toTensor();
  auto norm_type = inputs[2].toScalar();
  float eps = 1e-6;
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();
  auto num_params = static_cast<unsigned int>(gradients.size());

  auto slice_op = make_operator<SliceOperator>(device_id, scalar_type);

  if (norm_type.toFloat() == 2.0) {
    torch::jit::Stack stack;
    std::vector<Tensor> cat_input;
    auto cat_grad_norms = make_operator<CatOperator>(device_id, scalar_type);
    std::vector<int64_t> shape{1, 1};
    for (unsigned int i = 0; i < num_params; i++) {
      // Add node to compute norm on each gradient tensor
      auto norm_lp = make_operator<NormOperator>(device_id, scalar_type);
      norm_lp->SetSynapseInput(p_context_->syn_inputs_[i]);
      stack.emplace_back(IValue(gradients.get(i)));
      stack.emplace_back(IValue(2.0));
      norm_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
      // each grad_norm connected to cat node
      cat_input.push_back(norm_lp->GetOutputs()[0]);
      cat_grad_norms->SetSynapseInput(norm_lp->GetSynOutputs()[0]);
      stack.clear();
    }

    // grad_norms are concatened into a single big tensor of shape
    // {num_params,1}
    stack.emplace_back(IValue(cat_input));
    stack.emplace_back(IValue(0));
    cat_grad_norms->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // node to do compute total_norm
    auto norm_final = make_operator<NormOperator>(device_id, scalar_type);
    norm_final->SetSynapseInput(cat_grad_norms->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cat_grad_norms->GetOutputs()[0]));
    stack.emplace_back(IValue(2.0));
    norm_final->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();
    p_context_->syn_outputs_.emplace_back(
        std::move(norm_final->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(norm_final->GetOutputs()[0]));

    /*
    Now use the total_norm calculated to update grads
    max_norm = float(max_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in parameters:
        p.grad.detach().mul_(clip_coef)
    */

    // total_norm + 1e-6
    auto add_op1 = make_operator<AddOperator>(device_id, scalar_type);
    add_op1->SetSynapseInput(p_context_->syn_outputs_[0]);
    stack.emplace_back(IValue(p_context_->pt_outputs_[0]));
    stack.emplace_back(IValue(Scalar(eps)));
    stack.emplace_back(IValue(1.0));
    add_op1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // clip_coef = max_norm / (total_norm + 1e-6)
    auto div_final = make_operator<DivOperator>(device_id, scalar_type);
    div_final->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    div_final->SetSynapseInput(add_op1->GetSynOutputs()[0]);
    stack.emplace_back(IValue(max_grad_norm));
    stack.emplace_back(IValue(add_op1->GetOutputs()[0]));
    div_final->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // mask = total_norm > max_grad_norm
    auto gt_op = make_operator<GtOperator>(device_id, scalar_type);
    gt_op->SetSynapseInput(p_context_->syn_outputs_[0]);
    gt_op->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    stack.emplace_back(IValue(p_context_->pt_outputs_[0]));
    stack.emplace_back(IValue(max_grad_norm));
    gt_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    std::string node_type = "cast_i8_to_f32";
    auto cast1 = make_operator<CastOperator>(device_id, node_type);
    cast1->SetSynapseInput(gt_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(gt_op->GetOutputs()[0]));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    cast1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // mul1 = mask * clip_coef
    auto mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(cast1->GetSynOutputs()[0]);
    mul1->SetSynapseInput(div_final->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(div_final->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    // imask = (mask == 0)
    auto eq_op = make_operator<EqOperator>(device_id, scalar_type);
    eq_op->SetSynapseInput(cast1->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(0.0));
    eq_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    node_type = "cast_i8_to_f32";
    auto cast2 = make_operator<CastOperator>(device_id, node_type);
    cast2->SetSynapseInput(eq_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(eq_op->GetOutputs()[0]));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    cast2->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // mask*clip_coef + imask
    auto add_op2 = make_operator<AddOperator>(device_id, scalar_type);
    add_op2->SetSynapseInput(mul1->GetSynOutputs()[0]);
    add_op2->SetSynapseInput(cast2->GetSynOutputs()[0]);
    stack.emplace_back(IValue(mul1->GetOutputs()[0]));
    stack.emplace_back(IValue(cast2->GetOutputs()[0]));
    stack.emplace_back(IValue(1.0));
    add_op2->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
    // take just the first element of add since all values would be repeated

    slice_op->SetSynapseInput(add_op2->GetSynOutputs()[0]);
    stack.emplace_back(IValue(add_op2->GetOutputs()[0]));
    stack.emplace_back(IValue(0));
    stack.emplace_back(IValue(0));
    stack.emplace_back(IValue(1));
    stack.emplace_back(IValue(1));
    slice_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
  } else {
    // other norm_type not supported for now
    // BERT Hugging-face uses norm_type = 2.0
    // therefore supporting only that for now.
    HABANA_ASSERT(0, "unsupported norm_type for fused norm");
  }

  return slice_op;
}

void FusedNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for FusedNorm Operator");

  auto gradients = inputs[0].toTensorList();
  auto num_params = static_cast<unsigned int>(gradients.size());
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  auto slice_op = compute_clip_coeff(graph, inputs, output_metadata);

  torch::jit::Stack stack;

  // p.grad.detach().mul_(clip_coef)
  for (unsigned int i = 0; i < num_params; i++) {
    auto mul1 = make_operator<MulInplaceOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(p_context_->syn_inputs_[i]);
    mul1->SetSynapseInput(slice_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(slice_op->GetOutputs()[0]));
    auto out_metadata = SelectVectorIndices(output_metadata, {i + 1u});
    mul1->AllocateAndAddSynapseNode(graph, stack, out_metadata);
    stack.clear();
    // Add grads to output lists to satisfy GC (since grad updation is
    // inplace)
    p_context_->syn_outputs_.emplace_back(std::move(mul1->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(mul1->GetOutputs()[0]);
  }
}

void FusedNormLazyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for FusedNorm Operator");

  auto gradients = inputs[0].toTensorList();
  auto num_params = static_cast<unsigned int>(gradients.size());
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  auto slice_op = compute_clip_coeff(graph, inputs, output_metadata);

  torch::jit::Stack stack;

  // clipped_grad = p.grad.detach().mul(clip_coef)
  for (unsigned int i = 0; i < num_params; i++) {
    auto mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(p_context_->syn_inputs_[i]);
    mul1->SetSynapseInput(slice_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(slice_op->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(
        graph, stack, SelectVectorIndices(output_metadata, {i + 1}));
    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(mul1->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(mul1->GetOutputs()[0]);
  }
}

Tensor fused_norm_hpu(
    std::vector<Tensor>& grad,
    const Tensor& max_norm_t,
    float norm_type = 2.0) {
  PT_KERNEL_BEGIN;

  size_t device_id = grad[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = grad[0].scalar_type();
  std::string node_type =
      "fused_norm_" + habana_helpers::name_suffix_from_type(scalar_type);
  FusedNormOperator Op(device_id, scalar_type);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(grad), IValue(max_norm_t), IValue(norm_type)};

  std::vector<at::Tensor> pt_inputs;
  auto num_params = static_cast<int>(grad.size());
  pt_inputs.reserve(num_params);
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(grad[j]);
  }
  pt_inputs.push_back(max_norm_t);

  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = habana_helpers::createPTTensor(
        grad[0],
        grad[0].sizes().vec(),
        grad[0].options(),
        grad[0].suggest_memory_format(),
        true);
    std::vector<at::Tensor> pt_outputs;
    pt_outputs.push_back(output);
    for (auto j = 0; j < num_params; j++) {
      pt_outputs.push_back(grad[j]);
    }
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(pt_outputs);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    OutputMetaDataVector output_metadata(num_params + 1);
    for (auto& md : output_metadata) {
      md.persistent = true;
    }
    Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(
      out.size() == ((unsigned)num_params + 1), "Incorrect size of outputs");
  SET_SIZE_STRIDE_1D(out.at(0));

  PT_KERNEL_END;
  return out[0];
}

std::vector<int64_t> InstanceNormOperator::compute_output_shape(
    at::Tensor input,
    c10::MemoryFormat mf) {
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    auto is_norm_3d = is_5d_tensor(input.sizes().vec());
    auto channels_idx = is_norm_3d ? synapse_helpers::layouts::INPUT_3D_C_IDX
                                   : synapse_helpers::layouts::INPUT_C_IDX;
    return {input.sizes().vec()[0], input.sizes().vec()[channels_idx]};
  }
  // fetch channel dimension based on memory format
  auto is_norm_3d = is_5d_tensor(input.sizes().vec());
  constexpr int nchw_idx = 1;
  if (is_norm_3d) {
    auto nhwc_idx = 4;

    TORCH_CHECK(
        mf != c10::MemoryFormat::ChannelsLast,
        "Memory format should be ChannelsLast3d/Contiguous for 3d norm");

    auto channels_idx =
        (mf == c10::MemoryFormat::ChannelsLast3d) ? nhwc_idx : nchw_idx;

    return {input.sizes().vec()[0], input.sizes().vec()[channels_idx]};
  } else {
    auto nhwc_idx = 3;

    TORCH_CHECK(
        mf != c10::MemoryFormat::ChannelsLast3d,
        "Memory format should be ChannelsLast/Contiguous for 2d norm");

    auto channels_idx =
        (mf == c10::MemoryFormat::ChannelsLast) ? nhwc_idx : nchw_idx;

    return {input.sizes().vec()[0], input.sizes().vec()[channels_idx]};
  }
}

void InstanceNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(in_stack[3].isDouble(), "Input type expected to be double");
  TORCH_CHECK(
      output_metadata.size() == 3,
      "InstanceNormOperator: output_metadata should be 3 in training mode");

  auto input = in_stack[0].toTensor();

  auto is_norm_3d = is_5d_tensor(input.sizes().vec());

  kernel_meta_data_.synapse_input_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  kernel_meta_data_.synapse_output_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN});

  std::string guid = "instance_norm_fwd_" +
      habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  auto beta = in_stack[1].toTensor();

  const auto eps = in_stack[3].toDouble();

  auto output =
      habana_helpers::createPTTensor(input, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  auto memory_format = is_norm_3d ? c10::MemoryFormat::ChannelsLast3d
                                  : c10::MemoryFormat::ChannelsLast;
  auto mean_var_shape =
      InstanceNormOperator::compute_output_shape(input, memory_format);

  auto current_mean = habana_helpers::createPTTensor(
      beta,
      mean_var_shape,
      beta.options(),
      c10::MemoryFormat::Contiguous,
      output_metadata.at(1).persistent);

  auto current_istd = habana_helpers::createPTTensor(
      beta,
      mean_var_shape,
      beta.options(),
      c10::MemoryFormat::Contiguous,
      output_metadata.at(2).persistent);

  AllocateSynapseOutputs(
      graph,
      {current_mean, current_istd},
      {output_metadata.at(1), output_metadata.at(2)});

  // Note: TPC kernel doesnt support running mean and variance computation. we
  // just pass random momentum value as a place holder
  struct ns_InstanceNormTrainingKernel::Params params {
    0.9, static_cast<float>(eps)
  };

  p_context_->params_.emplace<ns_InstanceNormTrainingKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::vector<int64_t> InstanceNormBackwardOperator::compute_output_shape(
    at::Tensor input,
    c10::MemoryFormat mf) {
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    auto is_norm_3d = is_5d_tensor(input.sizes().vec());
    auto channels_idx = is_norm_3d ? synapse_helpers::layouts::INPUT_3D_C_IDX
                                   : synapse_helpers::layouts::INPUT_C_IDX;
    return {input.sizes().vec()[channels_idx]};
  }
  // fetch channel dimension based on memory format
  auto is_norm_3d = is_5d_tensor(input.sizes().vec());
  constexpr int nchw_idx = 1;

  if (is_norm_3d) {
    auto nhwc_idx = 4;

    TORCH_CHECK(
        mf != c10::MemoryFormat::ChannelsLast,
        "Memory format should be ChannelsLast3d/Contiguous for 3d norm");

    auto channels_idx =
        (mf == c10::MemoryFormat::ChannelsLast3d) ? nhwc_idx : nchw_idx;

    return {input.sizes().vec()[channels_idx]};
  } else {
    auto nhwc_idx = 3;

    TORCH_CHECK(
        mf != c10::MemoryFormat::ChannelsLast3d,
        "Memory format should be ChannelsLast/Contiguous for 2d norm");

    auto channels_idx =
        (mf == c10::MemoryFormat::ChannelsLast) ? nhwc_idx : nchw_idx;

    return {input.sizes().vec()[channels_idx]};
  }
}

void InstanceNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      output_metadata.size() == 3,
      "InstanceNormOperator: output_metadata should be 3 in training mode");
  auto input = in_stack[0].toTensor();
  auto mean = in_stack[2].toTensor();

  auto is_norm_3d = is_5d_tensor(input.sizes().vec());

  kernel_meta_data_.synapse_input_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN,
       synapse_helpers::layouts::SynapseLayoutFormat::CN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  kernel_meta_data_.synapse_output_layout.assign(
      {is_norm_3d ? synapse_helpers::layouts::SynapseLayoutFormat::WHDCN
                  : synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});

  std::string guid = "instance_norm_bwd_" +
      habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  auto output =
      habana_helpers::createPTTensor(input, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  auto memory_format = is_norm_3d ? c10::MemoryFormat::ChannelsLast3d
                                  : c10::MemoryFormat::ChannelsLast;
  auto grad_beta_gamma_shape =
      InstanceNormBackwardOperator::compute_output_shape(input, memory_format);

  auto grad_beta = habana_helpers::createPTTensor(
      mean,
      grad_beta_gamma_shape,
      mean.options(),
      c10::MemoryFormat::Contiguous,
      output_metadata.at(1).persistent);

  auto grad_gamma = habana_helpers::createPTTensor(
      mean,
      grad_beta_gamma_shape,
      mean.options(),
      c10::MemoryFormat::Contiguous,
      output_metadata.at(2).persistent);

  AllocateSynapseOutputs(
      graph,
      {grad_beta, grad_gamma},
      {output_metadata.at(1), output_metadata.at(2)});

  // Note: TPC kernel doesnt support running mean and variance computation. we
  // just pass random momentum value as a place holder
  struct ns_InstanceNormTrainingKernel::Params params {
    0.9, 1e-5
  };

  p_context_->params_.emplace<ns_InstanceNormTrainingKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

////////////////////////////////////////New BN

/*********************************************************************************
 * @brief - This op is used in lazy mode to avoid memcopy nodes for RMV
 *********************************************************************************/
at::Tensor BatchNormForwardOperator::create_or_return_tensor_bn(
    synapse_helpers::graph& graph,
    const Tensor& input,
    uint size,
    Device device,
    int syn_index) {
  Tensor ret_tensor;
  if (!input.defined()) {
    // If optional tensor is undefined, we create one and synapse tensor
    // appended info is patching info passed back to the kernel for this new
    // tensor which lowering kernel is unaware of
    ret_tensor = at::empty({size}, device);
    auto syn_tensor = habana_helpers::create_tensor(
        ret_tensor, graph, true, false, c10::nullopt);
    auto it = p_context_->syn_inputs_.begin() + syn_index;
    p_context_->syn_inputs_.insert(it, std::move(syn_tensor));

    appended_tensor_infos.emplace_back(
        std::make_tuple(syn_tensor.name(), ret_tensor, syn_tensor.id()));
  } else if (input.defined() && input.device() != DeviceType::HPU) {
    ret_tensor = input.to(DeviceType::HPU);
  } else {
    return input;
  }

  return ret_tensor;
}
at::Tensor BatchNormForwardOperator::create_or_return_pt_tensor_bn(
    const at::Tensor& input,
    uint size,
    Device device) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
  } else if (input.defined() && input.device() != DeviceType::HPU) {
    ret_tensor = input.to(DeviceType::HPU);
  } else {
    return input;
  }
  return ret_tensor;
}

void BatchNormForwardOperator::preProcessInputs(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect number of inputs against expected count for BatchNormForward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[5].isBool(), "Input type expected to be bool");
  TORCH_CHECK(inputs[6].isDouble(), "Input type expected to be double");
  TORCH_CHECK(inputs[7].isDouble(), "Input type expected to be double");

  const auto input = inputs[0].toTensor();
  const auto weight = inputs[1].toTensor();
  const auto bias = inputs[2].toTensor();
  const auto running_mean = inputs[3].toTensor();
  const auto running_var = inputs[4].toTensor();
  const auto training = inputs[5].toBool();
  TORCH_CHECK(
      training, "BN Forward training flag should be set to 1 in training mode");

  Tensor wt_hpu, bias_hpu;
  auto device = DeviceType::HPU;
  auto channel_dim = synapse_helpers::layouts::INPUT_C_IDX;

  if (training) {
    wt_hpu = create_or_return_tensor_bn(
        graph, weight, input.sizes()[channel_dim], device, (uint)1);
    bias_hpu = create_or_return_tensor_bn(
        graph, bias, input.sizes()[channel_dim], device, (uint)2);
  } else {
    bias_hpu = create_or_return_tensor_bn(
        graph, bias, input.sizes()[channel_dim], device, (uint)1);
    wt_hpu = create_or_return_tensor_bn(
        graph, weight, input.sizes()[channel_dim], device, (uint)2);
  }

  Tensor running_mean_hpu = create_or_return_tensor_bn(
      graph, running_mean, input.sizes()[channel_dim], device, (uint)3);
  Tensor running_var_hpu = create_or_return_tensor_bn(
      graph, running_var, input.sizes()[channel_dim], device, (uint)4);

  pre_inputs = {
      std::move(input),
      std::move(wt_hpu),
      std::move(bias_hpu),
      std::move(running_mean_hpu),
      std::move(running_var_hpu)};
  pt_inputs = {
      pre_inputs[0],
      pre_inputs[1],
      pre_inputs[2],
      pre_inputs[3],
      pre_inputs[4]};

  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
}

void BatchNormForwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    const OutputMetaDataVector& output_metadata) {
  preProcessInputs(graph, in_stack);
  TORCH_CHECK(
      output_metadata.size() == 5,
      "BatchNormForwardOperator: output_metadata should be 5 in training mode");
  const auto training = in_stack[5].toBool();
  const auto momentum = in_stack[6].toDouble();
  const auto eps = in_stack[7].toDouble();
  auto output = habana_helpers::createPTTensor(
      pre_inputs[0], output_metadata.at(0).persistent);

  AllocateSynapseOutput(graph, output, output_metadata.at(0));

  auto current_mean = habana_helpers::createPTTensor(
      pre_inputs[3], output_metadata.at(1).persistent);
  auto current_istd = habana_helpers::createPTTensor(
      pre_inputs[4], output_metadata.at(2).persistent);
  OutputMetaDataVector out_12_metadata =
      SelectVectorIndices(output_metadata, {1, 2});
  AllocateSynapseOutputs(graph, {current_mean, current_istd}, out_12_metadata);

  auto running_mean_out = habana_helpers::createPTTensor(
      pre_inputs[3], output_metadata.at(3).persistent);
  auto running_var_out = habana_helpers::createPTTensor(
      pre_inputs[4], output_metadata.at(4).persistent);
  OutputMetaDataVector out_34_metadata =
      SelectVectorIndices(output_metadata, {3, 4});
  AllocateSynapseOutputs(
      graph, {running_mean_out, running_var_out}, out_34_metadata);

  ns_BatchNormKernel::ParamsV2 params;
  params.momentum = static_cast<float>(momentum),
  params.epsilon = static_cast<float>(eps);
  params.threshold.f = 0.0;
  params.isTraining = training;
  p_context_->params_.emplace<ns_BatchNormKernel::ParamsV2>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

OutputShapeInfRetType BatchNormForwardOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  const auto input = inputs[0].toTensor();
  const auto running_mean = inputs[3].toTensor();
  const auto running_var = inputs[4].toTensor();

  OutputShapeInfRetType out;
  out.AddOutputTensor(TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));

  auto mean_tensor_data = TensorMetaData(
      running_mean.sizes().vec(),
      HabanaOperator::CalculateStrides(
          running_mean.sizes(), running_mean.suggest_memory_format()),
      running_mean.scalar_type(),
      running_mean.suggest_memory_format());

  auto var_tensor_data = TensorMetaData(
      running_var.sizes().vec(),
      HabanaOperator::CalculateStrides(
          running_var.sizes(), running_var.suggest_memory_format()),
      running_var.scalar_type(),
      running_var.suggest_memory_format());

  // current_mean and current_istd
  out.AddOutputTensor(mean_tensor_data);
  out.AddOutputTensor(var_tensor_data);

  // running_mean and running_var
  out.AddOutputTensor(mean_tensor_data);
  out.AddOutputTensor(var_tensor_data);

  return out;
}

// BN New Backward
void BatchNormBackwardOperator::create_opt_input_tensor_bn_bwd(
    synapse_helpers::graph& graph,
    const Tensor& input,
    uint size,
    Device device,
    int pos) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
    auto syn_tensor = habana_helpers::create_tensor(
        ret_tensor, graph, true, false, c10::nullopt);
    appended_tensor_infos.emplace_back(
        std::make_tuple(syn_tensor.name(), ret_tensor, syn_tensor.id()));
    // if input is not defined, we get dummy tensor from wrapper
    // create new one and place it to the original position
    p_context_->syn_inputs_.emplace_back(std::move(syn_tensor));
    if ((uint)pos < p_context_->pt_inputs_.size()) {
      p_context_->pt_inputs_[pos] = ret_tensor;
    } else {
      // this is for bias which is not present in input list
      // pushed to the end
      p_context_->pt_inputs_.push_back(ret_tensor);
    }
  }
}

void BatchNormBackwardOperator::preProcessInputs(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect number of inputs against expected count for BatchNormBackwardOperator preProcessInputs: expected 5 got ",
      inputs.size());
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");

  const auto input = inputs[0].toTensor();
  const auto mean = inputs[2].toTensor();
  const auto invstd = inputs[3].toTensor();
  const auto weight = inputs[4].toTensor();

  auto device = input.device();
  auto channel_dim = synapse_helpers::layouts::INPUT_C_IDX;

  create_opt_input_tensor_bn_bwd(
      graph, mean, input.sizes()[channel_dim], device, 2);
  create_opt_input_tensor_bn_bwd(
      graph, invstd, input.sizes()[channel_dim], device, 3);
  create_opt_input_tensor_bn_bwd(
      graph, weight, input.sizes()[channel_dim], device, 4);

  SetProprocessingDone();
}

void BatchNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect number of inputs against expected count for BatchNormBackwardOperator AllocateAndAddSynapseNode");
  TORCH_CHECK(
      inputs[6].isDouble(), "Input type for eps is expected to be double");
  TORCH_CHECK(
      output_metadata.size() == 3,
      "BatchNormBackwardOperator: #output_metadata should be 3");
  if (CheckProprocessingDone() == false) {
    Stack preprocess_in = {};
    for (auto& input : inputs) {
      if (input.isTensor()) {
        preprocess_in.insert(preprocess_in.end(), input);
      }
    }
    preProcessInputs(graph, preprocess_in);
  }
  const auto input = inputs[0].toTensor();
  const auto weight = inputs[4].toTensor();
  const auto training = inputs[5].toBool();
  const auto eps = inputs[6].toDouble();
  const auto momentum = inputs[7].toDouble();
  // Prepare output tensor vector
  auto grad_in_nhwc =
      habana_helpers::createPTTensor(input, output_metadata.at(0).persistent);
  auto grad_beta =
      habana_helpers::createPTTensor(weight, output_metadata.at(1).persistent);
  auto grad_gamma =
      habana_helpers::createPTTensor(weight, output_metadata.at(2).persistent);

  AllocateSynapseOutputs(
      graph, {grad_in_nhwc, grad_beta, grad_gamma}, output_metadata);

  ns_BatchNormKernel::ParamsV2 params;
  params.momentum = static_cast<float>(momentum),
  params.epsilon = static_cast<float>(eps);
  params.threshold.f = 0.0;
  params.isTraining = training;
  p_context_->params_.emplace<ns_BatchNormKernel::ParamsV2>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

OutputShapeInfRetType BatchNormBackwardOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  const auto input = inputs[0].toTensor();
  const auto weight = inputs[4].toTensor();

  OutputShapeInfRetType out;
  // grad_in_nhwc
  out.AddOutputTensor(TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  // grad_gamma
  out.AddOutputTensor(TensorMetaData(
      weight.sizes().vec(),
      HabanaOperator::CalculateStrides(
          weight.sizes(), weight.suggest_memory_format()),
      weight.scalar_type(),
      weight.suggest_memory_format()));
  // grad_beta
  out.AddOutputTensor(TensorMetaData(
      weight.sizes().vec(),
      HabanaOperator::CalculateStrides(
          weight.sizes(), weight.suggest_memory_format()),
      weight.scalar_type(),
      weight.suggest_memory_format()));
  return out;
}

///////////////////////////////////////
static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::native_batch_norm_training",
            KERNEL_FN(BatchNormForwardOperator))
        .add("hpu::native_batch_norm_inf", KERNEL_FN(BatchNormInfOperator))
        .add(
            "hpu::native_batch_norm_backward",
            KERNEL_FN(BatchNormBackwardOperator))
        .add("hpu::fused_norm_", KERNEL_FN(FusedNormOperator))
        .add("hpu::fused_norm_lazy", KERNEL_FN(FusedNormLazyOperator))
        .add("aten::native_layer_norm", KERNEL_FN(LayerNormOperator))
        .add(
            "aten::native_layer_norm_backward",
            KERNEL_FN(LayerNormBackwardOperator))
        .add("aten::norm.Scalar", KERNEL_FN(NormOperator))
        .add("aten::norm.ScalarOpt_dim", KERNEL_FN(NormOperator))
        .add("aten::norm.ScalarOpt_dim_dtype", KERNEL_FN(NormOperator))
        .add("hpu::instance_norm", KERNEL_FN(InstanceNormOperator))
        .add(
            "hpu::instance_norm_backward",
            KERNEL_FN(InstanceNormBackwardOperator));
