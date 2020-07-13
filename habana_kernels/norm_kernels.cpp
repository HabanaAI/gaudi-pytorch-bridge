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

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "norm_kernels.h"

using namespace torch;
/**********************************************************************
*@brief Changes dimensions of the input tensor as per specified dimension.
*This is done by adding dummy x1 dimensions. Eg: NC -> NCHW is done by
* resizing to NxCx1x1. Resizing is done inplace
@param input - Tensor, 2D/3D/4D, float/bf16
@param num_out_dim - uint, Range ~[2,4]
***********************************************************************/

Tensor batch_norm_resize(
    const Tensor& input,
    uint num_out_dim,
    c10::MemoryFormat memory_format) {
  auto num_in_dim = input.dim();
  Tensor input_resize = at::alias(input);

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
      } else {
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
      } else {
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
    output = input.to(DeviceType::HABANA);
  } else {
    output = at::empty(
        {size}, TensorOptions().dtype(c10::ScalarType::Float).device(device));
  }
  return output;
}

std::vector<at::Tensor> BatchNormForwardOperator::preProcessInputs(
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

  Tensor wt_hpu =
      get_batch_norm_optional_tensors(weight, input.sizes()[3], input.device());

  Tensor bias_hpu =
      get_batch_norm_optional_tensors(bias, input.sizes()[3], input.device());

  Tensor running_mean_hpu = get_batch_norm_optional_tensors(
      running_mean, input.sizes()[3], input.device());

  Tensor running_var_hpu = get_batch_norm_optional_tensors(
      running_var, input.sizes()[3], input.device());
  Tensor running_mean_hpu_in, running_var_hpu_in, residualAdd;
  std::vector<const at::Tensor*> pt_inputs;
  if (training == true) {
    if (running_mean.defined()) {
      running_mean_hpu_in =
          at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());

      running_var_hpu_in =
          at::empty(running_var_hpu.sizes(), running_var_hpu.options());

      residualAdd = at::empty(input.sizes(), input.options());

      // This residual add is dummy tensor to match the API requirements

      // running mean and running var cannot be in input and output list
      // simultaneously. create a copy
      habana_helpers::copy_data_within_device(
          running_mean_hpu, running_mean_hpu_in);

      habana_helpers::copy_data_within_device(
          running_var_hpu, running_var_hpu_in);
    }
    return {std::move(input),
            std::move(wt_hpu),
            std::move(bias_hpu),
            std::move(residualAdd),
            std::move(running_mean_hpu),
            std::move(running_var_hpu),
            std::move(running_mean_hpu_in),
            std::move(running_var_hpu_in)};
  } else {
    // training=false - Evaluation mode
    return {std::move(input),
            std::move(bias_hpu),
            std::move(wt_hpu),
            std::move(running_mean_hpu),
            std::move(running_var_hpu)};
  }
}

void BatchNormForwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 6,
      "Incorrect number of inputs against expected count for BatchNormForward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isBool(), "Input type expected to be bool");
  TORCH_CHECK(inputs[4].isDouble(), "Input type expected to be double");
  TORCH_CHECK(inputs[5].isDouble(), "Input type expected to be double");
  const auto input = inputs[0].toTensor();
  const auto running_mean_hpu = inputs[1].toTensor();
  const auto running_var_hpu = inputs[2].toTensor();
  const auto training = inputs[3].toBool();
  const auto momentum = inputs[4].toDouble();
  const auto eps = inputs[5].toDouble();

  auto output = at::empty(input.sizes(), input.options());
  auto current_mean =
      at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());
  auto current_istd =
      at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());

  if (training == true) {
    // synapse uses expAvgfactor = 1 - momentum
    struct synCudBnExParams params = {synBnOps::BN_OPS_BN,
                                      static_cast<float>(1 - momentum),
                                      static_cast<float>(eps)};
    p_context_->params_.emplace<synCudBnExParams>(params);
    p_context_->params_size_ = sizeof(params);
    AllocateSynapseOutputs(
        graph,
        {output, running_mean_hpu, running_var_hpu, current_mean, current_istd},
        is_output_persistent);
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  } else {
    struct ns_BatchNormKernel::Params params;
    params.threshold.f = 0.0;
    params.momentum = static_cast<float>(momentum);
    params.epsilon = static_cast<float>(eps);
    p_context_->params_.emplace<ns_BatchNormKernel::Params>(params);
    p_context_->params_size_ = sizeof(params);
    AllocateSynapseOutput(graph, output, is_output_persistent);
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void BatchNormForwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  const auto input = inputs[0].toTensor();
  const auto running_mean_hpu = inputs[1].toTensor();
  const auto running_var_hpu = inputs[2].toTensor();
  const auto training = inputs[3].toBool();
  // Prepare output tensor vector
  auto output = at::empty(input.sizes(), input.options());
  if (training == true) {
    auto current_mean =
        at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());
    auto current_istd =
        at::empty(running_mean_hpu.sizes(), running_mean_hpu.options());
    HabanaOperator::SetPTOutputs({output,
                                  running_mean_hpu,
                                  running_var_hpu,
                                  current_mean,
                                  current_istd});
  } else {
    HabanaOperator::SetPTOutput(output);
  }
}
/*******************************************************************
*@brief Implements forward pass for batch norm
*INPUTS
@param input - IFM, 2D/3D/4D, bf16/FP32, NHWC
@param weight - Gamma in PyT, 1D, FP32, C (optional)
@param bias - beta in PyT, 1D, FP32, C (optional)
@param running_mean - Filtered mean 1D, FP32, C (optional)
@param running_var - Filtered variance, 1D, FP32, C (optional)
@param momentum - Update factor, float
@param eps - float (for numerical stability during
divisions)
*OUTPUTS
@param output - OFM, 2D/3D/4D, bf16/FP32, NHWC
@param running_mean_out - Updated filtered mean 1D, FP32, C
@param running_var_out - Updated filtered variance 1D, FP32, C
*UNUSED variables will be enabled later once evaluation/inference mode is
*implemented
*******************************************************************/
std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  PT_KERNEL_BEGIN;
  // Build Params for the graph
  Stack cache_stack = {IValue(input),
                       IValue(weight),
                       IValue(bias),
                       IValue(running_mean),
                       IValue(running_var),
                       IValue(training),
                       IValue(momentum),
                       IValue(eps)};
  auto num_input_dim = input.dim();
  TORCH_CHECK(num_input_dim > 1, "Expected range of input dimensions is [2,4]");
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  // Resize input to 4D to match TPC kernel requirement.
  auto input_resize = batch_norm_resize(input, 4, memory_format);
  Tensor input_nhwc = input_resize;
  std::vector<const at::Tensor*> pt_in = {&input_resize};
  std::vector<at::Tensor*> pt_out = {&input_nhwc};
  IntArrayRef new_dim_pos = {0, 2, 3, 1};
  std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  // The following tensors are autogenerated by pytorch. Hence need to be
  // pushed to device first. The below operations can be removed in graph
  // mode when the intermediate tensors lives in the device
  // AFAIK, in training mode, all the optional tensors are received.
  // Nevertheless, if we don't receive the optional tensors from PyT, create
  // empty ones to satify TPC kernel input constraints

  size_t device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type = training ? "cud_bn_fwd_ex" : "batch_norm_inf";
  auto batch_norm = [&] {
    // Create the operator
    BatchNormForwardOperator Op(device_id, scalar_type, node_type);

    // Build Params for the graph
    Stack preprocess_stack = {IValue(input_nhwc),
                              IValue(weight),
                              IValue(bias),
                              IValue(running_mean),
                              IValue(running_var),
                              IValue(training),
                              IValue(momentum),
                              IValue(eps)};
    // Assign Inputs to the Operator
    auto pre_inputs = Op.preProcessInputs(preprocess_stack);
    std::vector<const at::Tensor*> pt_inputs;
    // Build Params for the graph
    Stack input_stack;
    if (training == true) {
      pt_inputs = {&pre_inputs[0],
                   &pre_inputs[1],
                   &pre_inputs[2],
                   &pre_inputs[3],
                   &pre_inputs[6],
                   &pre_inputs[7]};
      input_stack = {IValue(input_nhwc),
                     IValue(pre_inputs[4]),
                     IValue(pre_inputs[5]),
                     IValue(training),
                     IValue(momentum),
                     IValue(eps)};
    } else {
      pt_inputs = {&pre_inputs[0],
                   &pre_inputs[1],
                   &pre_inputs[2],
                   &pre_inputs[3],
                   &pre_inputs[4]};
      input_stack = {IValue(input_nhwc),
                     IValue(pre_inputs[3]),
                     IValue(pre_inputs[4]),
                     IValue(training),
                     IValue(momentum),
                     IValue(eps)};
    }
    size_t key = Op.GetRecipeKey(node_type, cache_stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Stack out_stack;
      if (training) {
        out_stack = {IValue(input_nhwc),
                     IValue(pre_inputs[4]),
                     IValue(pre_inputs[5]),
                     IValue(training)};
      } else {
        out_stack = {IValue(input_nhwc),
                     IValue(pre_inputs[3]),
                     IValue(pre_inputs[4]),
                     IValue(training)};
      }
      Op.SetPTOutputs(out_stack);
      Op.Execute(key);
    } else {
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      Op.AllocateSynapseInputs(graph, pt_inputs, true);

      Op.AllocateAndAddSynapseNode(graph, input_stack, true);

      Op.Compile(graph);
    }

    std::vector<at::Tensor> out = Op.GetOutputs();
    return out;
  };

  auto bn_outputs = batch_norm();
  auto output_nhwc = bn_outputs[0];
  Tensor output = output_nhwc;
  pt_in = {&output_nhwc};
  pt_out = {&output};
  new_dim_pos = {0, 3, 1, 2};
  pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Resize output
  auto output_resized = batch_norm_resize(output, num_input_dim, memory_format);

  PT_KERNEL_END;
  if (training == false) {
    // for eval, return mean and var are not used and we can return anything
    // give back any tensor of same shape
    Tensor running_mean_hpu = get_batch_norm_optional_tensors(
        running_mean, input.sizes()[3], input.device());

    Tensor running_var_hpu = get_batch_norm_optional_tensors(
        running_var, input.sizes()[3], input.device());
    return std::make_tuple(output_resized, running_mean_hpu, running_var_hpu);
  }
  return std::make_tuple(output_resized, bn_outputs[3], bn_outputs[4]);
}

std::vector<at::Tensor> BatchNormBackwardOperator::preProcessInputs(
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect number of inputs against expected count for BatchNormBackward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  const auto input = inputs[0].toTensor();
  const auto grad_out = inputs[1].toTensor();
  const auto weight = inputs[2].toTensor();
  const auto save_mean = inputs[3].toTensor();
  const auto save_invstd = inputs[4].toTensor();

  Tensor wt_hpu =
      get_batch_norm_optional_tensors(weight, input.sizes()[3], input.device());
  Tensor bias_hpu = at::zeros(wt_hpu.sizes(), wt_hpu.options().memory_format(wt_hpu.suggest_memory_format()));
  Tensor save_mean_hpu = get_batch_norm_optional_tensors(
      save_mean, input.sizes()[3], input.device());
  Tensor save_invstd_hpu = get_batch_norm_optional_tensors(
      save_invstd, input.sizes()[3], input.device());
  return {std::move(input),
          std::move(grad_out),
          std::move(wt_hpu),
          std::move(bias_hpu),
          std::move(save_mean_hpu),
          std::move(save_invstd_hpu)};
}

void BatchNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect number of inputs against expected count for BatchNormBackward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isDouble(), "Input type expected to be double");
  const auto input = inputs[0].toTensor();
  const auto weight = inputs[1].toTensor();
  const auto eps = inputs[2].toDouble();

  // Prepare output tensor vector
  auto grad_in_nhwc = at::empty(input.sizes(), input.options());
  auto grad_beta = at::empty(weight.sizes(), weight.options());
  auto grad_gamma = at::empty(weight.sizes(), weight.options());

  struct synCudBnExParams params = {
      synBnOps::BN_OPS_BN, 0, static_cast<float>(eps)};
  p_context_->params_.emplace<synCudBnExParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutputs(
      graph, {grad_in_nhwc, grad_gamma, grad_beta}, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void BatchNormBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  const auto input = inputs[0].toTensor();
  const auto wt_hpu = inputs[1].toTensor();
  // Prepare output tensor vector
  auto grad_in_nhwc = at::empty(input.sizes(), input.options());
  auto grad_beta = at::empty(wt_hpu.sizes(), wt_hpu.options());
  auto grad_gamma = at::empty(wt_hpu.sizes(), wt_hpu.options());
  HabanaOperator::SetPTOutputs({grad_in_nhwc, grad_gamma, grad_beta});
}
/*******************************************************************
*@brief Implements backward pass for batch norm
*INPUTS
@param grad_out - output gradient tensor, 2D/3D/4D, bf16/FP32, NHWC
@param input - IFM, 2D/3D/4D, bf16/FP32, NHWC
@param weight - Gamma in PyT, 1D, FP32, C (optional)
@param running_mean - Filtered mean 1D, FP32, C (optional)
@param running_var - Filtered variance, 1D, FP32, C (optional)
@param save_mean - saved mean from fwd pass, 1D, FP32, C
@param save_invstd - saved inverse variance from fwd pass, 1D, FP32, C

*OUTPUTS
@param grad_in - input gradient tensor, 2D/3D/4D, bf16/FP32, NHWC
@param grad_gamma - gradient of weight 1D, FP32, C
@param grad_beta - gradient of bias 1D, FP32, C
*UNUSED variables will be enabled later once evaluation/inference mode is
*implemented
*******************************************************************/

std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu(
    Tensor& grad_out,
    Tensor& input,
    Tensor& weight,
    UNUSED Tensor& running_mean,
    UNUSED Tensor& running_var,
    Tensor& save_mean,
    Tensor& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask) {
  PT_KERNEL_BEGIN;
  bool output_mask_in[3];
  output_mask_in[0] = output_mask[0];
  output_mask_in[1] = output_mask[1];
  output_mask_in[2] = output_mask[2];
  // Build Params for the graph
  Stack cache_stack = {IValue(grad_out),
                       IValue(input),
                       IValue(weight),
                       IValue(running_mean),
                       IValue(running_var),
                       IValue(save_mean),
                       IValue(save_invstd),
                       IValue(train),
                       IValue(eps),
                       IValue(output_mask_in)};
  auto num_input_dim = input.dim();
  TORCH_CHECK(num_input_dim > 1, "Expected range of input dimensions is [2,4]");
  TORCH_CHECK(
      num_input_dim == grad_out.dim(),
      "Grad out dimension not matching that of input");
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  // Resize input and grad in to 4D to match TPC kernel requirement.
  auto input_resize = batch_norm_resize(input, 4, memory_format);
  auto grad_out_resize = batch_norm_resize(grad_out, 4, memory_format);

  Tensor input_nhwc = input_resize;
  Tensor grad_out_nhwc = grad_out_resize;
  std::vector<const at::Tensor*> pt_in{&input_resize, &grad_out_resize};
  std::vector<at::Tensor*> pt_out{&input_nhwc, &grad_out_nhwc};
  IntArrayRef new_dim_pos = {0, 2, 3, 1};
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos, &new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  size_t device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type = "cud_bn_bwd_ex";
  auto batch_norm_bwd = [&] {
    // Create the operator
    BatchNormBackwardOperator Op(device_id, scalar_type, node_type);

    // Build Params for the graph
    Stack preprocess_stack = {IValue(input_nhwc),
                              IValue(grad_out_nhwc),
                              IValue(weight),
                              IValue(save_mean),
                              IValue(save_invstd)};
    auto pre_inputs = Op.preProcessInputs(preprocess_stack);
    std::vector<const at::Tensor*> pt_inputs = {&pre_inputs[0],
                                                &pre_inputs[1],
                                                &pre_inputs[2],
                                                &pre_inputs[3],
                                                &pre_inputs[4],
                                                &pre_inputs[5]};

    size_t key = Op.GetRecipeKey(node_type, cache_stack);
    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.SetPTInputs(pt_inputs);
      Stack out_stack = {IValue(pre_inputs[0]), IValue(pre_inputs[2])};
      Op.SetPTOutputs(out_stack);
      Op.Execute(key);
    } else {
      PT_KERNEL_DEBUG("Key:", key);
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);

      Op.AllocateSynapseInputs(graph, pt_inputs, true);
      // Build Params for the graph
      Stack input_stack = {
          IValue(pre_inputs[0]), IValue(pre_inputs[2]), IValue(eps)};
      Op.AllocateAndAddSynapseNode(graph, input_stack, true);
      Op.Compile(graph);
    }

    std::vector<at::Tensor> out = Op.GetOutputs();
    return out;
  };
  auto bn_outputs = batch_norm_bwd();
  auto grad_in_nhwc = bn_outputs[0];
  Tensor grad_in = grad_in_nhwc;
  pt_in = {&grad_in_nhwc};
  pt_out = {&grad_in};
  new_dim_pos = {0, 3, 1, 2};
  pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Resize output
  auto grad_in_resized =
      batch_norm_resize(grad_in, num_input_dim, memory_format);

  PT_KERNEL_END;
  return std::make_tuple(grad_in_resized, bn_outputs[1], bn_outputs[2]);
}

/** @brief This function implements forward pass for torch.nn.LayerNorm()
 * @param input (bf16/fp32 tensor) input tensor
 * @param weight (fp32 tensor) per element scale value tensor
 * @param bias (fp32 tensor) per element bias value tensor
 * @param m (int) num of elements in outer dims not used in LayerNorm
 * @param n (int) num of elements used for computing LayerNorm
 * @param eps (double) a value added to the denominator for numerical stability.
 * Default: 1e-5
 */
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps) {
  PT_KERNEL_BEGIN;

  auto wt_reshaped = weight.view(-1);
  auto bias_reshaped = bias.view(-1);
  std::vector<int64_t> shape{m, n};
  auto input_reshaped = input.view(shape);

  std::vector<const at::Tensor*> pt_inputs{
      &input_reshaped, &bias_reshaped, &wt_reshaped};

  std::vector<int64_t> shape_mean{m, 1};
  IntArrayRef meanArray(shape_mean.data(), shape_mean.size());
  auto output = at::empty(input_reshaped.sizes(), input_reshaped.options());
  auto mean = at::empty(meanArray, wt_reshaped.options());
  auto istd = at::empty(meanArray, bias_reshaped.options());

  std::vector<const at::Tensor*> pt_outputs{&output, &mean, &istd};

  struct ns_LayerNormKernel::Params param;
  param.eps = static_cast<float>(eps);
  param.epsValid = true;

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "layer_norm",
      &param,
      sizeof(param),
      SynapsePassType::FORWARD_PASS);

  auto output_reshaped = output.view(input.sizes().vec());

  PT_KERNEL_END;
  return std::make_tuple(
      std::move(output_reshaped), std::move(mean), std::move(istd));
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

  TORCH_CHECK(p.toFloat() > 0.0, "norm with p > 0.0 is only supported");

  auto self_hpu = self.view(-1);
  auto output = at::empty(self_hpu.sizes(), self.options(), self.suggest_memory_format());
  auto retain = at::empty(self_hpu.sizes(), self.options(), self.suggest_memory_format());

  ns_LpNormKernel::Params params{};
  params.p = p.to<float>();
  params.dim = 0;
  params.eps = 1e-5;

  std::vector<const at::Tensor*> pt_inputs{&self_hpu};
  std::vector<const at::Tensor*> pt_outputs{&output, &retain};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "lpnorm",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  at::reciprocal_(retain);

  // PT expects 0-D
  retain.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  PT_KERNEL_END;
  return retain;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(batch_norm_hpu),
                    &batch_norm_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(batch_norm_bwd_hpu),
                    &batch_norm_bwd_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)")
                .impl_unboxedOnlyKernel<
                    decltype(layer_norm_hpu),
                    &layer_norm_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(norm_scalar_hpu),
                    &norm_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
