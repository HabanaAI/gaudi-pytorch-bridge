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

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/unary_kernels.h"

using namespace torch;
/**********************************************************************
*@brief Changes dimensions of the input tensor as per specified dimension.
*This is done by adding dummy x1 dimensions. Eg: NC -> NCHW is done by
* resizing to NxCx1x1. Resizing is done inplace
@param input - Tensor, 2D/3D/4D, float/bf16
@param num_out_dim - uint, Range ~[2,4]
***********************************************************************/

void BatchNormForwardOperator::remove_non_persistent_patching_info() {
  size_t del_idx = 0;
  size_t loop_size = p_context_->syn_inputs_.size();
  for (size_t i = 0; i < loop_size; i++) {
    synapse_helpers::tensor& syn_in = p_context_->syn_inputs_[del_idx];
    if (syn_in.is_persistent() == false) {
      auto it = p_context_->syn_inputs_.begin() + del_idx;
      p_context_->syn_inputs_.erase(it);
      auto itv = p_context_->pt_inputs_.begin() + del_idx;
      p_context_->pt_inputs_.erase(itv);
    } else {
      del_idx++;
    }
  }
}

Tensor batch_norm_resize(
    const Tensor& input,
    uint num_out_dim,
    c10::MemoryFormat memory_format) {
  auto num_in_dim = input.dim();
  Tensor input_resize = at::alias(input.to(DeviceType::HABANA));

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
    return input.to(DeviceType::HABANA);
  } else {
    output = at::empty(
        {size}, TensorOptions().dtype(c10::ScalarType::Float).device(device));
  }
  return output;
}

void BatchNormForwardOperator::insert_memcopy_op(
    synapse_helpers::graph& graph,
    at::Tensor& src,
    at::Tensor& dst,
    int in_position) {
  MemCopyOperator memcopyOp(this->p_context_->device_id_, this->scalarType_);
  auto& syn_temp = memcopyOp.SetSynapseInput(
      std::move(p_context_->syn_inputs_[in_position]));
  // No need for output PT tensor as its non persistent
  torch::jit::Stack stack = {IValue(src), IValue(dst)};
  memcopyOp.AllocateAndAddSynapseNode(graph, stack, false);
  synapse_helpers::tensor& syn_tensor = memcopyOp.GetSynOutputs()[0];
  mean_var_temp.emplace_back(std::move(syn_temp));
  p_context_->syn_inputs_[in_position] = std::move(syn_tensor);
}

at::Tensor BatchNormForwardOperator::create_or_return_pt_tensor_bn(
    const at::Tensor& input,
    uint size,
    Device device) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
  } else if (input.defined() && input.device() != DeviceType::HABANA) {
    ret_tensor = input.to(DeviceType::HABANA);
  } else {
    return input;
  }
  return ret_tensor;
}
at::Tensor BatchNormForwardOperator::create_or_return_tensor_bn(
    synapse_helpers::graph& graph,
    const Tensor& input,
    uint size,
    Device device,
    int syn_index) {
  Tensor ret_tensor;
  if (!input.defined()) {
    // If optiona tensor is undefined, we create one and synapse tensor
    // appended info is patching info passed back to the kernel for this new
    // tensor which lowering kernel is unaware of
    ret_tensor = at::empty({size}, device);
    auto syn_tensor = habana_helpers::create_tensor(
        ret_tensor, graph.get_graph_handle(), true, c10::nullopt);
    auto it = p_context_->syn_inputs_.begin() + syn_index;
    p_context_->syn_inputs_.insert(it, std::move(syn_tensor));

    appended_tensor_infos.emplace_back((syn_tensor).tensor_name_, ret_tensor);
  } else if (input.defined() && input.device() != DeviceType::HABANA) {
    ret_tensor = input.to(DeviceType::HABANA);
    ;
  } else {
    return input;
  }

  return ret_tensor;
}

void BatchNormForwardOperator::generateCacheInputs(Stack& inputs) {
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
  const auto momentum = inputs[6].toDouble();
  const auto eps = inputs[7].toDouble();

  std::string guid = training ? "cud_bn_fwd_ex"
                              : "batch_norm_inf_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  Tensor wt_hpu, bias_hpu;
  auto device = DeviceType::HABANA;

  if (training) {
    wt_hpu = create_or_return_pt_tensor_bn(weight, input.sizes()[3], device);
    bias_hpu = create_or_return_pt_tensor_bn(bias, input.sizes()[3], device);
  } else {
    bias_hpu = create_or_return_pt_tensor_bn(bias, input.sizes()[3], device);
    wt_hpu = create_or_return_pt_tensor_bn(weight, input.sizes()[3], device);
  }

  running_vars_def = running_mean.defined();
  Tensor running_mean_hpu =
      create_or_return_pt_tensor_bn(running_mean, input.sizes()[3], device);
  Tensor running_var_hpu =
      create_or_return_pt_tensor_bn(running_var, input.sizes()[3], device);

  Tensor residualAdd;
  if (training == true) {
    // Mean and Var syn tensors may be reused as they are not bound to data
    // create a new syn tensor for bias and add it
    // This residual add is dummy tensor to match the API requirements
    residualAdd = at::empty(input.sizes(), input.options());
    pre_inputs = {std::move(input),
                  std::move(wt_hpu),
                  std::move(bias_hpu),
                  std::move(residualAdd),
                  std::move(running_mean_hpu),
                  std::move(running_var_hpu)};
    // Mean and var only passed once as intermediate ones are non-persistent
    // adn dont go for patching
    pt_inputs = {pre_inputs[0],
                 pre_inputs[1],
                 pre_inputs[2],
                 pre_inputs[3],
                 pre_inputs[4],
                 pre_inputs[5]};
    input_stack = {IValue(pre_inputs[0]),
                   IValue(pre_inputs[4]),
                   IValue(pre_inputs[5]),
                   IValue(training),
                   IValue(momentum),
                   IValue(eps)};
  } else {
    // training=false - Evaluation mode
    pre_inputs = {std::move(input),
                  std::move(bias_hpu),
                  std::move(wt_hpu),
                  std::move(running_mean_hpu),
                  std::move(running_var_hpu)};
    pt_inputs = {pre_inputs[0],
                 pre_inputs[1],
                 pre_inputs[2],
                 pre_inputs[3],
                 pre_inputs[4]};
    input_stack = {IValue(pre_inputs[0]),
                   IValue(pre_inputs[3]),
                   IValue(pre_inputs[4]),
                   IValue(training),
                   IValue(momentum),
                   IValue(eps)};
  }
  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
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
  const auto momentum = inputs[6].toDouble();
  const auto eps = inputs[7].toDouble();

  std::string guid = training ? "cud_bn_fwd_ex"
                              : "batch_norm_inf_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  Tensor wt_hpu, bias_hpu;
  auto device = DeviceType::HABANA;

  if (training) {
    wt_hpu = create_or_return_tensor_bn(
        graph, weight, input.sizes()[3], device, (uint)1);
    bias_hpu = create_or_return_tensor_bn(
        graph, bias, input.sizes()[3], device, (uint)2);
  } else {
    bias_hpu = create_or_return_tensor_bn(
        graph, bias, input.sizes()[3], device, (uint)1);
    wt_hpu = create_or_return_tensor_bn(
        graph, weight, input.sizes()[3], device, (uint)2);
  }

  running_vars_def = running_mean.defined();
  Tensor running_mean_hpu = create_or_return_tensor_bn(
      graph, running_mean, input.sizes()[3], device, (uint)3);
  Tensor running_var_hpu = create_or_return_tensor_bn(
      graph, running_var, input.sizes()[3], device, (uint)4);

  Tensor residualAdd;
  if (training == true) {
    // AS we dont support in place ops, transfer mean and var data
    // to new non-persistent tensor and use that as input
    // The original tensors are used as outputs.
    // As the intermediate tensor is non persistent, they dont need PT Tensor
    // So we reuse the original PT tensor(for meta data)
    if (running_vars_def) {
      insert_memcopy_op(graph, running_mean_hpu, running_mean_hpu, 3);
      insert_memcopy_op(graph, running_var_hpu, running_var_hpu, 4);
    }
    // Mean and Var syn tensors may be reused as they are not bound to data
    // create a new syn tensor for bias and add it
    // This residual add is dummy tensor to match the API requirements
    residualAdd = at::empty(input.sizes(), input.options());
    auto syn_tensor_add = habana_helpers::create_tensor(
        residualAdd, graph.get_graph_handle(), true, c10::nullopt);
    auto it = p_context_->syn_inputs_.begin() + 3;

    // This is to communicate to graph lowering that a new tensor was
    // added by the kernel and it can add to patching in lowering
    appended_tensor_infos.emplace_back(
        (syn_tensor_add).tensor_name_, residualAdd);
    p_context_->syn_inputs_.insert(it, std::move(syn_tensor_add));

    pre_inputs = {std::move(input),
                  std::move(wt_hpu),
                  std::move(bias_hpu),
                  std::move(residualAdd),
                  std::move(running_mean_hpu),
                  std::move(running_var_hpu)};
    pt_inputs = {pre_inputs[0],
                 pre_inputs[1],
                 pre_inputs[2],
                 pre_inputs[3],
                 pre_inputs[4],
                 pre_inputs[5],
                 pre_inputs[4],
                 pre_inputs[5]};
    input_stack = {IValue(pre_inputs[0]),
                   IValue(pre_inputs[4]),
                   IValue(pre_inputs[5]),
                   IValue(training),
                   IValue(momentum),
                   IValue(eps)};
  } else {
    // training=false - Evaluation mode
    pre_inputs = {std::move(input),
                  std::move(bias_hpu),
                  std::move(wt_hpu),
                  std::move(running_mean_hpu),
                  std::move(running_var_hpu)};
    pt_inputs = {pre_inputs[0],
                 pre_inputs[1],
                 pre_inputs[2],
                 pre_inputs[3],
                 pre_inputs[4]};
    input_stack = {IValue(pre_inputs[0]),
                   IValue(pre_inputs[3]),
                   IValue(pre_inputs[4]),
                   IValue(training),
                   IValue(momentum),
                   IValue(eps)};
  }
  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
}

void BatchNormForwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    std::vector<bool> is_output_persistent) {
  // Add intermediate tensors and add to op
  preProcessInputs(graph, in_stack);

  TORCH_CHECK(in_stack[5].isBool(), "Input type expected to be bool");
  TORCH_CHECK(in_stack[6].isDouble(), "Input type expected to be double");
  TORCH_CHECK(in_stack[7].isDouble(), "Input type expected to be double");
  TORCH_CHECK(
      is_output_persistent.size() == 3,
      "BatchNormForwardOperator: is_output_persistent should be 3");
  const auto training = in_stack[5].toBool();
  const auto momentum = in_stack[6].toDouble();
  const auto eps = in_stack[7].toDouble();

  auto output =
      habana_helpers::createPTTensor(pre_inputs[0], is_output_persistent[0]);

  if (training == true) {
    // synapse uses expAvgfactor = 1 - momentum
    struct synCudBnExParams params = {synBnOps::BN_OPS_BN,
                                      static_cast<float>(1 - momentum),
                                      static_cast<float>(eps)};
    p_context_->params_.emplace<synCudBnExParams>(params);
    p_context_->params_size_ = sizeof(params);

    AllocateSynapseOutput(graph, output, is_output_persistent[0]);

    auto current_mean =
        habana_helpers::createPTTensor(pre_inputs[4], is_output_persistent[1]);
    auto current_istd =
        habana_helpers::createPTTensor(pre_inputs[5], is_output_persistent[2]);
    // Intermediate tensors are non-persistent
    // Modifit the PT tensors too to not allocate mem
    if (running_vars_def) {
      // As the tensors are used as IO and are persistent
      // WE need to use same mem section
      auto syn_tensor_mean = habana_helpers::duplicate_tensor_in_memory_section(
          (mean_var_temp[0]));
      auto syn_tensor_var = habana_helpers::duplicate_tensor_in_memory_section(
          (mean_var_temp[1]));

      appended_tensor_infos.emplace_back(
          (syn_tensor_mean).tensor_name_, pre_inputs[4]);
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_mean));

      appended_tensor_infos.emplace_back(
          (syn_tensor_var).tensor_name_, pre_inputs[5]);
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_var));
    } else {
      // As the tensors are used as IO and are persistent
      // WE need to use same mem section
      auto syn_tensor_mean = habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[3]);
      auto syn_tensor_var = habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[4]);

      appended_tensor_infos.emplace_back(
          (syn_tensor_mean).tensor_name_, pre_inputs[4]);
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_mean));

      appended_tensor_infos.emplace_back(
          (syn_tensor_var).tensor_name_, pre_inputs[5]);
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_var));
    }

    p_context_->pt_outputs_.emplace_back(pre_inputs[4]);
    p_context_->pt_outputs_.emplace_back(pre_inputs[5]);

    std::vector<bool> persistent_output_flags{is_output_persistent[1],
                                              is_output_persistent[2]};
    AllocateSynapseOutputs(
        graph, {current_mean, current_istd}, persistent_output_flags);

    AddNodeToSynapseGraph(graph, &params, sizeof(params));
    // We need to put original syn tensors for mean and Var in patching
    // Although ununsed in graph, they are still accounted for by GC
    p_context_->syn_inputs_.emplace_back(std::move(mean_var_temp[0]));
    p_context_->syn_inputs_.emplace_back(std::move(mean_var_temp[1]));
    p_context_->excluded_output_indices_.insert(1);
    p_context_->excluded_output_indices_.insert(2);
  } else {
    struct ns_BatchNormKernel::Params params;
    params.threshold.f = 0.0;
    params.momentum = static_cast<float>(momentum);
    params.epsilon = static_cast<float>(eps);
    p_context_->params_.emplace<ns_BatchNormKernel::Params>(params);
    p_context_->params_size_ = sizeof(params);
    AllocateSynapseOutput(graph, output, is_output_persistent[0]);
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
    // for eval, return mean and var are not used and we can return anything
    // give back any tensor of same shape
    p_context_->pt_outputs_.emplace_back(pre_inputs[3]);
    p_context_->pt_outputs_.emplace_back(pre_inputs[4]);
  }
  remove_non_persistent_patching_info();
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
    HabanaOperator::SetPTOutputs({output, running_mean_hpu, running_var_hpu});
  }
}

std::vector<at::Tensor>& BatchNormForwardOperator::GetBNInputs() {
  return pt_inputs;
}

torch::jit::Stack& BatchNormForwardOperator::GetInputstack() {
  return input_stack;
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
  int64_t pos[] = {0, 2, 3, 1};
  IntArrayRef new_dim_pos = pos;
  std::vector<const IntArrayRef*> pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  Stack in_stack = {IValue(input_nhwc),
                    IValue(weight),
                    IValue(bias),
                    IValue(running_mean),
                    IValue(running_var),
                    IValue(training),
                    IValue(momentum),
                    IValue(eps)};
  // The following tensors are autogenerated by pytorch. Hence need to be
  // pushed to device first. The below operations can be removed in graph
  // mode when the intermediate tensors lives in the device
  // AFAIK, in training mode, all the optional tensors are received.
  // Nevertheless, if we don't receive the optional tensors from PyT, create
  // empty ones to satify TPC kernel input constraints

  size_t device_id = input.device().index();

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type = "bn_fwd";
  auto batch_norm = [&] {
    // Create the operator
    BatchNormForwardOperator Op(device_id, scalar_type);
    size_t key = Op.GetRecipeKey(node_type, cache_stack);
    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Op.generateCacheInputs(in_stack);
      Op.SetPTOutputs(Op.GetInputstack());
      Op.Execute(key);
    } else {
      PT_KERNEL_DEBUG("key:", key);
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      for (auto ival : in_stack) {
        if (ival.isTensor() && ival.toTensor().defined()) {
          at::Tensor in = ival.toTensor().to(DeviceType::HABANA);
          Op.AllocateSynapseInput(graph, in, true);
        }
      }
      Op.AllocateAndAddSynapseNode(graph, in_stack, {true, true, true});

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
  int64_t new_pos[] = {0, 3, 1, 2};
  new_dim_pos = new_pos;
  pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Resize output
  auto output_resized = batch_norm_resize(output, num_input_dim, memory_format);

  PT_KERNEL_END;
  if (training == false) {
    return std::make_tuple(output_resized, bn_outputs[1], bn_outputs[2]);
  }
  return std::make_tuple(output_resized, bn_outputs[3], bn_outputs[4]);
}

Tensor BatchNormBackwardOperator::create_or_return_input_tensor_bn_bwd(
    synapse_helpers::graph& graph,
    const Tensor& input,
    uint size,
    Device device,
    int syn_index) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
    auto syn_tensor = habana_helpers::create_tensor(
        ret_tensor, graph.get_graph_handle(), true, c10::nullopt);
    reordered_syn_inputs_.emplace_back(std::move(syn_tensor));

    appended_tensor_infos.emplace_back((syn_tensor).tensor_name_, ret_tensor);
  } else if (input.defined() && input.device() != DeviceType::HABANA) {
    reordered_syn_inputs_.emplace_back(
        std::move(p_context_->syn_inputs_[syn_index]));
    ret_tensor = input.to(DeviceType::HABANA);
  } else {
    return input;
  }

  return ret_tensor;
}

at::Tensor BatchNormBackwardOperator::create_or_return_pt_tensor_bn(
    const Tensor& input,
    uint size,
    Device device) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
  } else if (input.defined() && input.device() != DeviceType::HABANA) {
    ret_tensor = input.to(DeviceType::HABANA);
  } else {
    return input;
  }
  return ret_tensor;
}

void BatchNormBackwardOperator::preProcessInputs(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect number of inputs against expected count for BatchNormBackward preProcessInputs: expected 7 got ",
      inputs.size());
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  const auto grad_out = inputs[0].toTensor();
  const auto input = inputs[1].toTensor();
  const auto weight = inputs[2].toTensor();
  const auto running_mean = inputs[3].toTensor();
  const auto running_var = inputs[4].toTensor();
  const auto save_mean = inputs[5].toTensor();
  const auto save_invstd = inputs[6].toTensor();
  bool running_mean_def = running_mean.defined();
  bool running_var_def = running_var.defined();
  unsigned int running_mean_pos;
  unsigned int running_var_pos;
  std::vector<synapse_helpers::tensor_or_ref> mean_var_temp;
  if (running_mean_def) {
    running_mean_pos = weight.defined() ? 3 : 2;
    mean_var_temp.emplace_back(
        std::move(p_context_->syn_inputs_[running_mean_pos]));
  } else {
    running_mean_pos = 0; // not defined - not used
  }
  if (running_var_def) {
    if (running_mean_def) {
      running_var_pos = weight.defined() ? 4 : 3;
      mean_var_temp.emplace_back(
          std::move(p_context_->syn_inputs_[running_var_pos]));
    } else {
      running_var_pos = weight.defined() ? 3 : 2;
      mean_var_temp.emplace_back(
          std::move(p_context_->syn_inputs_[running_var_pos]));
    }
  } else {
    running_var_pos = 0; // not defined - not used
  }

  reordered_syn_inputs_.clear();
  Tensor wt_hpu, bias_hpu, save_mean_hpu, save_invstd_hpu;
  auto device = input.device();

  reordered_syn_inputs_.emplace_back(
      std::move(p_context_->syn_inputs_[1])); // input
  reordered_syn_inputs_.emplace_back(
      std::move(p_context_->syn_inputs_[0])); // grad_out
  wt_hpu = create_or_return_input_tensor_bn_bwd(
      graph, weight, input.sizes()[3], device, (int)2);
  // After 2nd input tensor, we add bias which is new
  bias_hpu = create_or_return_input_tensor_bn_bwd(
      graph, bias_hpu, input.sizes()[3], device, (int)3);
  save_mean_hpu = create_or_return_input_tensor_bn_bwd(
      graph,
      save_mean,
      input.sizes()[3],
      device,
      (int)5); // original save_mean pos is 5
  save_invstd_hpu = create_or_return_input_tensor_bn_bwd(
      graph,
      save_invstd,
      input.sizes()[3],
      device,
      (int)6); // original save_var pos is 6

  pre_inputs = {
      input, grad_out, wt_hpu, bias_hpu, save_mean_hpu, save_invstd_hpu};

  pt_inputs = {pre_inputs[0],
               pre_inputs[1],
               pre_inputs[2],
               pre_inputs[3],
               pre_inputs[4],
               pre_inputs[5]};
  p_context_->syn_inputs_.clear();
  for (auto& st : reordered_syn_inputs_) {
    // p_context_->syn_inputs_.emplace_back(std::move(st));
    SetSynapseInput(std::move(st));
  }
  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
  input_stack = {IValue(pre_inputs[0]),
                 IValue(pre_inputs[2])}; // for creating output tensors
  SetProprocessingDone();
}

void BatchNormBackwardOperator::swapGradInput() {
  std::swap(p_context_->syn_inputs_[0], p_context_->syn_inputs_[1]);
}
void BatchNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 10,
      "Incorrect number of inputs against expected count for BatchNormBackward AllocateAndAddSynapseNode");
  TORCH_CHECK(
      inputs[8].isDouble(), "Input type for eps is expected to be double");
  TORCH_CHECK(
      is_output_persistent.size() == 3,
      "BatchNormBackwardOperator: #is_output_persistent should be 3");
  if (CheckProprocessingDone() == false) {
    Stack preprocess_in = {};
    for (auto& input : inputs) {
      if (input.isTensor()) {
        preprocess_in.insert(preprocess_in.end(), input);
      }
    }
    preProcessInputs(graph, preprocess_in);
  }
  const auto input = pt_inputs[0];
  const auto weight = pt_inputs[2];
  const auto eps = inputs[8].toDouble();

  // Prepare output tensor vector
  auto grad_in_nhwc =
      habana_helpers::createPTTensor(input, is_output_persistent[0]);
  auto grad_beta =
      habana_helpers::createPTTensor(weight, is_output_persistent[1]);
  auto grad_gamma =
      habana_helpers::createPTTensor(weight, is_output_persistent[2]);

  struct synCudBnExParams params = {
      synBnOps::BN_OPS_BN, 0, static_cast<float>(eps)};
  p_context_->params_.emplace<synCudBnExParams>(params);
  p_context_->params_size_ = sizeof(params);
  AllocateSynapseOutputs(
      graph, {grad_in_nhwc, grad_gamma, grad_beta}, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  // swap input and grad for graph mode return
  swapGradInput();
}

void BatchNormBackwardOperator::generateCacheInputs(Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect number of inputs against expected count for BatchNormBackward generateCacheInputs: expected 7 got ",
      inputs.size());
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  const auto grad_out = inputs[0].toTensor();
  const auto input = inputs[1].toTensor();
  const auto weight = inputs[2].toTensor();
  const auto running_mean = inputs[3].toTensor();
  const auto running_var = inputs[4].toTensor();
  const auto save_mean = inputs[5].toTensor();
  const auto save_invstd = inputs[6].toTensor();

  Tensor wt_hpu, bias_hpu, save_mean_hpu, save_invstd_hpu;
  auto device = input.device();

  wt_hpu = create_or_return_pt_tensor_bn(weight, input.sizes()[3], device);
  // After 2nd input tensor, we add bias which is new
  bias_hpu = create_or_return_pt_tensor_bn(bias_hpu, input.sizes()[3], device);
  save_mean_hpu = create_or_return_pt_tensor_bn(
      save_mean,
      input.sizes()[3],
      device); // original save_mean pos is 5
  save_invstd_hpu = create_or_return_pt_tensor_bn(
      save_invstd,
      input.sizes()[3],
      device); // original save_var pos is 6

  pre_inputs = {
      input, grad_out, wt_hpu, bias_hpu, save_mean_hpu, save_invstd_hpu};

  pt_inputs = {pre_inputs[0],
               pre_inputs[1],
               pre_inputs[2],
               pre_inputs[3],
               pre_inputs[4],
               pre_inputs[5]};

  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
  input_stack = {IValue(pre_inputs[0]),
                 IValue(pre_inputs[2])}; // for creating output tensors
  SetProprocessingDone();
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
  int64_t dim_pos[] = {0, 2, 3, 1};
  IntArrayRef new_dim_pos = dim_pos;
  std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos, &new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);

  size_t device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type = "cud_bn_bwd_ex";
  Stack preprocess_stack = {IValue(grad_out_nhwc),
                            IValue(input_nhwc),
                            IValue(weight),
                            IValue(running_mean),
                            IValue(running_var),
                            IValue(save_mean),
                            IValue(save_invstd),
                            IValue(train),
                            IValue(eps),
                            IValue(output_mask_in)};
  auto batch_norm_bwd = [&] {
    // Create the operator
    BatchNormBackwardOperator Op(device_id, scalar_type);

    Op.SetResizeDone(); // true for eager mode
    size_t key = Op.GetRecipeKey(node_type, cache_stack);
    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Stack cache_preprocess_stack = {IValue(grad_out_nhwc),
                                      IValue(input_nhwc),
                                      IValue(weight),
                                      IValue(running_mean),
                                      IValue(running_var),
                                      IValue(save_mean),
                                      IValue(save_invstd)};
      Op.generateCacheInputs(cache_preprocess_stack);
      Op.SetPTOutputs(Op.GetInputstack());
      Op.Execute(key);
    } else {
      PT_KERNEL_DEBUG("Key:", key);
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      for (auto ival : preprocess_stack) {
        if (ival.isTensor() && ival.toTensor().defined()) {
          at::Tensor in = ival.toTensor().to(DeviceType::HABANA);
          Op.AllocateSynapseInput(graph, in, true);
        }
      }
      // Build Params for the graph
      Op.AllocateAndAddSynapseNode(graph, preprocess_stack, {true, true, true});
      // swap the grad and input again as patching table needs reversed ones
      Op.swapGradInput();
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
  int64_t new_dim_pos_arr[] = {0, 3, 1, 2};
  new_dim_pos = new_dim_pos_arr;
  pt_new_pos = {&new_dim_pos};
  habana_helpers::change_tensors_to_memory_format(
      pt_out, pt_in, pt_new_pos, memory_format);
  // Resize output
  auto grad_in_resized =
      batch_norm_resize(grad_in, num_input_dim, memory_format);

  PT_KERNEL_END;
  return std::make_tuple(grad_in_resized, bn_outputs[1], bn_outputs[2]);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
LayerNormOperator::getOutputSizes(const at::Tensor& input, int m) {
  auto output_sizes = input.sizes().vec();
  std::vector<int64_t> shape_mean{m, 1};
  return std::make_tuple(output_sizes, shape_mean, shape_mean);
}
std::tuple<Tensor, Tensor, Tensor> LayerNormOperator::AllocatePTOutputs(
    const Tensor& input,
    const Tensor& bias,
    const Tensor& weight,
    int64_t m,
    std::array<bool, 3> is_persistent) {
  std::vector<int64_t> shape_mean{m, 1};
  auto sizes = LayerNormOperator::getOutputSizes(input, m);
  auto output = habana_helpers::createPTTensor(
      input,
      std::get<0>(sizes),
      input.options(),
      input.suggest_memory_format(),
      is_persistent[0]);
  auto istd = habana_helpers::createPTTensor(
      bias,
      std::get<1>(sizes),
      bias.options(),
      bias.suggest_memory_format(),
      is_persistent[1]);
  auto mean = habana_helpers::createPTTensor(
      weight,
      std::get<2>(sizes),
      weight.options(),
      weight.suggest_memory_format(),
      is_persistent[2]);

  return std::make_tuple(std::move(output), std::move(mean), std::move(istd));
}

void LayerNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // For now calling the persistentce with single output
  AllocateAndAddSynapseNode(
      graph,
      inputs,
      {is_output_persistent, is_output_persistent, is_output_persistent});
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
void LayerNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 6,
      "LayerNormOperator::AllocateAndAddSynapseNode expected 6 args but got ",
      inputs.size())
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isInt(), "Input type expected to be int64");
  TORCH_CHECK(inputs[4].isInt(), "Input type expected to be int64");
  TORCH_CHECK(inputs[5].isDouble(), "Input type expected to be double");
  const auto input = inputs[0].toTensor();
  const auto weight = inputs[1].toTensor();
  const auto bias = inputs[2].toTensor();
  auto m = inputs[3].toInt();
  auto n = inputs[4].toInt();
  const auto eps = inputs[5].toDouble();

  //PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION Env variable is added as WA only for BERT graph mode 
  //and it should not be enabled in other cases.
  static const std::string graphFusionEnvValue = "PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION";
  const char* graphFusionValue = getenv(graphFusionEnvValue.c_str());
  if(graphFusionValue)
  {
    int isFusionEnabled = std::stoi(getenv("PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION"));
    if (isFusionEnabled && m==-1)
    {
      m = input.size(0) * input.size(1);
      if(n != input.size(2))
      {
        n = input.size(2);
      }
    }
  }

  std::vector<int64_t> shape_mean{m, 1};
  IntArrayRef meanArray(shape_mean.data(), shape_mean.size());

  // swap the pt and syn inputs for weight and bias to reflect the order in
  // which they have to be handed over to TPC kernel
  std::swap(p_context_->pt_inputs_[1], p_context_->pt_inputs_[2]);
  std::swap(p_context_->syn_inputs_[1], p_context_->syn_inputs_[2]);

  // Add Reshape node for input to graph for input.view({m,n})
  ReshapeOperator reshape_op_input(input.device().index(), input.scalar_type());
  auto& syn_in_input =
      reshape_op_input.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  int64_t modified_input_sizes[] = {m, n};
  c10::IntArrayRef modified_input_shape(modified_input_sizes, 2);
  torch::jit::Stack stack = {c10::IValue(input),
                             c10::IValue(modified_input_shape)};
  reshape_op_input.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[0] = std::move(syn_in_input);
  auto input_reshaped = reshape_op_input.GetOutputs()[0];
  synapse_helpers::tensor& syn_in_ln = reshape_op_input.GetSynOutputs()[0];

  // Add Reshape node for bias to graph for bias.view(-1)
  ReshapeOperator reshape_op_bias(bias.device().index(), bias.scalar_type());
  auto& syn_in_bias =
      reshape_op_bias.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  int64_t sizes[1];
  sizes[0] = bias.numel();
  c10::IntArrayRef modified_bias_shape(sizes, 1);
  stack = {c10::IValue(bias), c10::IValue(modified_bias_shape)};
  reshape_op_bias.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(syn_in_bias);
  auto bias_reshaped = reshape_op_bias.GetOutputs()[0];
  synapse_helpers::tensor& syn_bias_ln = reshape_op_bias.GetSynOutputs()[0];

  // Add Reshape node for weight to graph for weight.view(-1)
  ReshapeOperator reshape_op_wt(weight.device().index(), weight.scalar_type());
  auto& syn_in_wt =
      reshape_op_wt.SetSynapseInput(std::move(p_context_->syn_inputs_[2]));
  sizes[0] = weight.numel();
  c10::IntArrayRef modified_weight_shape(sizes, 1);
  stack = {c10::IValue(weight), c10::IValue(modified_weight_shape)};
  reshape_op_wt.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[2] = std::move(syn_in_wt);
  auto wt_reshaped = reshape_op_wt.GetOutputs()[0];
  synapse_helpers::tensor& syn_wt_ln = reshape_op_wt.GetSynOutputs()[0];

  auto outputs = AllocatePTOutputs(
      input,
      bias_reshaped,
      wt_reshaped,
      m,
      {false, is_output_persistent[1], is_output_persistent[2]});
  auto output = std::get<0>(outputs);
  auto mean = std::get<1>(outputs);
  auto istd = std::get<2>(outputs);
  std::vector<synTensor> syn_inputs{syn_in_ln.get()};
  syn_inputs.push_back(syn_bias_ln.get());
  syn_inputs.push_back(syn_wt_ln.get());
  // output syn tensor is non-persistent since it will be reshaped to input
  // sizes which will be marked as persistent
  AllocateSynapseOutput(
      graph, habana_helpers::createPTTensor(input_reshaped, false), false);
  AllocateSynapseOutput(graph, mean, is_output_persistent[1]);
  AllocateSynapseOutput(graph, istd, is_output_persistent[2]);
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
      std::move(guid_));
  // Add Reshape node for output tensor to graph -
  // output.view(input.sizes().vec())
  ReshapeOperator reshape_op_out(input.device().index(), input.scalar_type());
  reshape_op_out.SetSynapseInput(std::move(p_context_->syn_outputs_[0]));
  stack = {c10::IValue(input_reshaped), c10::IValue(input.sizes().vec())};
  reshape_op_out.AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[0]);
  synapse_helpers::tensor& syn_reshape_out = reshape_op_out.GetSynOutputs()[0];
  p_context_->syn_outputs_[0] = std::move(syn_reshape_out);
  p_context_->pt_outputs_[0] = reshape_op_out.GetOutputs()[0];
}

void LayerNormOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 6,
      "LayerNormOperator::AllocateAndAddSynapseNode expected 6 args but got ",
      inputs.size())
  const auto input = inputs[0].toTensor();
  const auto weight = inputs[1].toTensor();
  const auto bias = inputs[2].toTensor();
  auto m = inputs[3].toInt();

  //PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION Env variable is added as WA only for BERT graph mode 
  //and it should not be enabled in other cases.
  static const std::string graphFusionEnvValue = "PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION";
  const char* graphFusionValue = getenv(graphFusionEnvValue.c_str());
  if(graphFusionValue)
  {
    int isFusionEnabled = std::stoi(getenv("PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION"));
    if (isFusionEnabled && m==-1)
    {
      m = input.size(0) * input.size(1);
    }
  }

  auto outputs = AllocatePTOutputs(input, bias, weight, m, {true, true, true});
  HabanaOperator::SetPTOutputs(
      {std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)});
}

/** @brief This function implements forward pass for torch.nn.LayerNorm()
 * @param input (bf16/fp32 tensor) input tensor
 * @param weight (fp32 tensor) per element scale value tensor
 * @param bias (fp32 tensor) per element bias value tensor
 * @param m (int) num of elements in outer dims not used in LayerNorm
 * @param n (int) num of elements used for computing LayerNorm
 * @param eps (double) a value added to the denominator for numerical
 * stability. Default: 1e-5
 */
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps) {
  PT_KERNEL_BEGIN;
  // Build Params for the graph
  Stack input_stack = {IValue(input),
                       IValue(weight),
                       IValue(bias),
                       IValue(m),
                       IValue(n),
                       IValue(eps)};
  size_t device_id = input.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type = "layer_norm";
  std::vector<at::Tensor> out;

  auto layer_norm = [&] {
    // Create the operator
    LayerNormOperator Op(device_id, scalar_type);

    size_t key = Op.GetRecipeKey(node_type, input_stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      // Assign Inputs to the Operator
      const std::vector<at::Tensor> pt_inputs{input, bias, weight};
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(input_stack);
      Op.Execute(key);
      out = Op.GetOutputs();
    } else {
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      const std::vector<at::Tensor> pt_inputs{input, weight, bias};
      Op.AllocateSynapseInputs(graph, pt_inputs, true);
      Op.AllocateAndAddSynapseNode(graph, input_stack, true);
      Op.Compile(graph);
      out = Op.GetOutputs();
    }
    return out;
  };
  auto ln_outputs = layer_norm();

  PT_KERNEL_END;
  return std::make_tuple(
      std::move(ln_outputs[0]),
      std::move(ln_outputs[1]),
      std::move(ln_outputs[2]));
}

std::tuple<Tensor, Tensor, Tensor> LayerNormBackwardOperator::AllocatePTOutputs(
    const Tensor& input,
    const Tensor& weight,
    bool is_persistent) {
  auto sizes = LayerNormBackwardOperator::getOutputSizes(input, weight);
  auto output = habana_helpers::createPTTensor(
      input,
      std::get<0>(sizes),
      input.options(),
      input.suggest_memory_format(),
      is_persistent);
  auto beta = habana_helpers::createPTTensor(
      weight,
      std::get<1>(sizes),
      weight.options(),
      weight.suggest_memory_format(),
      is_persistent);
  auto gamma = habana_helpers::createPTTensor(
      weight,
      std::get<2>(sizes),
      weight.options(),
      weight.suggest_memory_format(),
      is_persistent);

  return std::make_tuple(std::move(output), std::move(beta), std::move(gamma));
}

void LayerNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // For now calling the persistentce with single output
  AllocateAndAddSynapseNode(
      graph,
      inputs,
      {is_output_persistent, is_output_persistent, is_output_persistent});
}

void LayerNormBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 8,
      "LayerNormBackwardOperator::AllocateAndAddSynapseNode expected 8 args but got ",
      inputs.size())
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[5].isInt(), "Input type expected to be int64");
  TORCH_CHECK(inputs[6].isInt(), "Input type expected to be int64");
  TORCH_CHECK(inputs[7].isBoolList(), "Input type expected to be bool array");

  const auto dY = inputs[0].toTensor();
  const auto X = inputs[1].toTensor();
  const auto mean = inputs[2].toTensor();
  const auto rstd = inputs[3].toTensor();
  const auto gamma = inputs[4].toTensor();
  auto m = inputs[5].toInt();
  auto n = inputs[6].toInt();
  const auto grad_input_mask = inputs[7].toBoolList();

  //PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION Env variable is added as WA only for BERT graph mode 
  //and it should not be enabled in other cases.
  static const std::string graphFusionEnvValue = "PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION";
  const char* graphFusionValue = getenv(graphFusionEnvValue.c_str());
  if(graphFusionValue)
  {
    int isFusionEnabled = std::stoi(getenv("PT_HABANA_ENABLE_GRAPHMODE_LAYERNORM_FUSION"));
    if (isFusionEnabled && m==-1)
    {
      m = X.size(0) * X.size(1);
      if(n != X.size(2))
      {
        n = X.size(2);
      }
    }
  }

  // swap the inputs for grad-in and input to reflect the order in
  // which they have to be handed over to TPC kernel
  std::swap(p_context_->pt_inputs_[0], p_context_->pt_inputs_[1]);
  std::swap(p_context_->syn_inputs_[0], p_context_->syn_inputs_[1]);

  // Add Reshape node for input to graph for input.view({m,n})
  ReshapeOperator reshape_op_x(X.device().index(), X.scalar_type());
  auto& syn_in_x =
      reshape_op_x.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  int64_t modified_x_sizes[] = {m, n};
  c10::IntArrayRef modified_x_shape(modified_x_sizes, 2);
  torch::jit::Stack stack = {c10::IValue(X), c10::IValue(modified_x_shape)};
  reshape_op_x.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[0] = std::move(syn_in_x);
  auto x_reshaped = reshape_op_x.GetOutputs()[0];
  synapse_helpers::tensor& syn_x = reshape_op_x.GetSynOutputs()[0];
  stack.clear();

  // Add Reshape node for input to graph for grad_in.view({m,n})
  ReshapeOperator reshape_op_dy(dY.device().index(), dY.scalar_type());
  auto& syn_in_dy =
      reshape_op_dy.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  int64_t modified_dy_sizes[] = {m, n};
  c10::IntArrayRef modified_dy_shape(modified_dy_sizes, 2);
  stack = {c10::IValue(dY), c10::IValue(modified_dy_shape)};
  reshape_op_dy.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(syn_in_dy);
  auto dy_reshaped = reshape_op_dy.GetOutputs()[0];
  synapse_helpers::tensor& syn_dy = reshape_op_dy.GetSynOutputs()[0];

  // Add Reshape node for weight to graph for weight.view(-1)
  ReshapeOperator reshape_op_gamma(gamma.device().index(), gamma.scalar_type());
  auto& syn_in_gamma =
      reshape_op_gamma.SetSynapseInput(std::move(p_context_->syn_inputs_[4]));
  int64_t sizes[1];
  sizes[0] = gamma.numel();
  c10::IntArrayRef modified_gamma_shape(sizes, 1);
  stack = {c10::IValue(gamma), c10::IValue(modified_gamma_shape)};
  reshape_op_gamma.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[4] = std::move(syn_in_gamma);
  auto gamma_reshaped = reshape_op_gamma.GetOutputs()[0];
  synapse_helpers::tensor& syn_gamma = reshape_op_gamma.GetSynOutputs()[0];

  // Add layer_norm_bwd node to graph
  std::vector<synTensor> syn_inputs{syn_x.get(), syn_dy.get()};
  synapse_helpers::tensor& syn_mean = p_context_->syn_inputs_[2];
  syn_inputs.push_back(syn_mean.get());
  synapse_helpers::tensor& syn_lstd = p_context_->syn_inputs_[3];
  syn_inputs.push_back(syn_lstd.get());
  syn_inputs.push_back(syn_gamma.get());
  // output syn tensors are non-persistent since these will be reshaped to
  // input sizes which will be marked as persistent
  auto outputs = AllocatePTOutputs(dY, gamma, false);
  auto output0 = std::get<0>(outputs);
  auto output1 = std::get<1>(outputs);
  auto output2 = std::get<2>(outputs);
  AllocateSynapseOutput(
      graph, habana_helpers::createPTTensor(dy_reshaped, false), false);
  AllocateSynapseOutput(
      graph, habana_helpers::createPTTensor(gamma_reshaped, false), false);
  AllocateSynapseOutput(
      graph, habana_helpers::createPTTensor(gamma_reshaped, false), false);
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
      std::move(guid_));

  // Add Reshape nodes for output tensors to graph
  ReshapeOperator reshape_op_grad_in(dY.device().index(), dY.scalar_type());
  reshape_op_grad_in.SetSynapseInput(std::move(p_context_->syn_outputs_[0]));
  stack = {c10::IValue(output0), c10::IValue(output0.sizes().vec())};
  reshape_op_grad_in.AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[0]);
  synapse_helpers::tensor& syn_reshape_grad_in =
      reshape_op_grad_in.GetSynOutputs()[0];

  ReshapeOperator reshape_op_grad_gamma(
      gamma.device().index(), gamma.scalar_type());
  reshape_op_grad_gamma.SetSynapseInput(std::move(p_context_->syn_outputs_[2]));
  stack = {c10::IValue(output2), c10::IValue(output2.sizes().vec())};
  reshape_op_grad_gamma.AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[2]);
  synapse_helpers::tensor& syn_reshape_grad_gamma =
      reshape_op_grad_gamma.GetSynOutputs()[0];

  ReshapeOperator reshape_op_grad_beta(
      gamma.device().index(), gamma.scalar_type());
  reshape_op_grad_beta.SetSynapseInput(std::move(p_context_->syn_outputs_[1]));
  stack = {c10::IValue(output1), c10::IValue(output1.sizes().vec())};
  reshape_op_grad_beta.AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[1]);
  synapse_helpers::tensor& syn_reshape_grad_beta =
      reshape_op_grad_beta.GetSynOutputs()[0];

  // beta & gamma are produced by TPC kernel in order opposite to
  // that required by PyTorch. Take care of swapping into correct
  // order here.
  p_context_->syn_outputs_[0] = std::move(syn_reshape_grad_in);
  p_context_->pt_outputs_[0] = reshape_op_grad_in.GetOutputs()[0];
  p_context_->syn_outputs_[1] = std::move(syn_reshape_grad_gamma);
  p_context_->pt_outputs_[1] = reshape_op_grad_gamma.GetOutputs()[0];
  p_context_->syn_outputs_[2] = std::move(syn_reshape_grad_beta);
  p_context_->pt_outputs_[2] = reshape_op_grad_beta.GetOutputs()[0];
}

void LayerNormBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  const auto dY = inputs[0].toTensor();
  const auto gamma = inputs[4].toTensor();

  auto outputs = AllocatePTOutputs(dY, gamma, true);
  HabanaOperator::SetPTOutputs(
      {std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)});
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
LayerNormBackwardOperator::getOutputSizes(
    const at::Tensor& input,
    const at::Tensor& gamma) {
  std::vector<int64_t> gamma_size =
      gamma.defined() ? gamma.sizes().vec() : input.sizes().vec();
  return std::make_tuple(input.sizes().vec(), gamma_size, gamma_size);
}
/** @brief This function implements backward pass for torch.nn.LayerNorm()
 * with grad_on_grad = False
 * @param dY (bf16/fp32 tensor) grad input tensor
 * @param X (bf16/fp32 tensor) input tensor
 * @param mean (fp32 tensor)
 * @param rstd (fp32 tensor)
 * @param gamma (fp32 tensor) per element scale value tensor
 * @param m (int) num of elements in outer dims not used in LayerNorm
 * @param n (int) num of elements used for computing LayerNorm
 * @param grad_input_mask (bool[]) mask to specify which output grads are
 * enabled
 */
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_hpu(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  PT_KERNEL_BEGIN;
  std::vector<bool> grad_mask_in;
  grad_mask_in.push_back(grad_input_mask[0]);
  grad_mask_in.push_back(grad_input_mask[1]);
  grad_mask_in.push_back(grad_input_mask[2]);

  // Build Params for the graph
  Stack input_stack = {IValue(dY),
                       IValue(X),
                       IValue(mean),
                       IValue(rstd),
                       IValue(gamma),
                       IValue(M),
                       IValue(N),
                       IValue(grad_mask_in)};
  size_t device_id = X.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = X.scalar_type();
  std::string node_type = "layer_norm";
  std::vector<at::Tensor> out;

  auto layer_norm = [&] {
    // Create the operator
    LayerNormBackwardOperator Op(device_id, scalar_type);

    size_t key = Op.GetRecipeKey(node_type, input_stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      // Assign Inputs to the Operator
      const std::vector<at::Tensor> pt_inputs{X, dY, mean, rstd, gamma};
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(input_stack);
      Op.Execute(key);
      out = Op.GetOutputs();
    } else {
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      const std::vector<at::Tensor> pt_inputs{dY, X, mean, rstd, gamma};
      Op.AllocateSynapseInputs(graph, pt_inputs, true);
      Op.AllocateAndAddSynapseNode(graph, input_stack, true);
      Op.Compile(graph);
      out = Op.GetOutputs();
    }
    return out;
  };
  auto ln_outputs = layer_norm();

  PT_KERNEL_END;
  return std::make_tuple(
      std::move(ln_outputs[0]),
      std::move(ln_outputs[1]),
      std::move(ln_outputs[2]));
}

void NormOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Norm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for Norm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for Norm Operator");

  auto self = inputs[0].toTensor();
  int64_t data[1];
  data[0] = self.numel();
  c10::IntArrayRef shape(data, 1);

  auto output = at::empty(shape.vec(), self.options(), c10::nullopt);
  HabanaOperator::SetPTOutput(output);
}
void NormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Norm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for Norm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for Norm Operator");

  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();

  // ReShape Operator
  at::ScalarType scalar_type = self.scalar_type();

  int64_t data[1];
  data[0] = self.numel();
  c10::IntArrayRef shape(data, 1);

  // Create the operator
  ReshapeOperator ReShapeOp(this->p_context_->device_id_, scalar_type);
  auto& reShape_syn =
      ReShapeOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

  // Build Params for the graph
  std::vector<c10::IValue> stack{IValue(self), IValue(shape)};
  ReShapeOp.AllocateAndAddSynapseNode(graph, stack, false);

  synapse_helpers::tensor& reshape_syn_tensor = ReShapeOp.GetSynOutputs()[0];
  auto output_reshape = ReShapeOp.GetOutputs()[0];
  p_context_->syn_inputs_[0] = std::move(reShape_syn);
  stack.clear();

  // LpNorm Operator
  // Create the operator
  LpNormOperator LpNormOp(this->p_context_->device_id_, scalar_type);
  LpNormOp.SetSynapseInput(std::move(reshape_syn_tensor));

  // Build Params for the graph
  stack.emplace_back(IValue(output_reshape));
  stack.emplace_back(IValue(p));
  LpNormOp.AllocateAndAddSynapseNode(graph, stack, {false, false});

  synapse_helpers::tensor& norm_syn_tensor = LpNormOp.GetSynOutputs()[1];
  auto output_norm = LpNormOp.GetOutputs()[1];
  stack.clear();

  // Reciprocal Operator
  // Create the operator
  ReciprocalOperator reciprocalOp(this->p_context_->device_id_, scalar_type);
  reciprocalOp.SetSynapseInput(std::move(norm_syn_tensor));

  // Build Params for the graph
  stack.emplace_back(IValue(output_norm));
  reciprocalOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

  synapse_helpers::tensor& reciprocal_syn_tensor =
      reciprocalOp.GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(std::move(reciprocal_syn_tensor));
  p_context_->pt_outputs_.emplace_back(reciprocalOp.GetOutputs()[0]);
}

void LpNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for LpNorm Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for LpNorm Operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be Scalar for LpNorm Operator");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "LpNormOperator: #is_output_persistent should be 2");

  auto self = inputs[0].toTensor();
  auto p = inputs[1].toScalar();

  TORCH_CHECK(p.toFloat() > 0.0, "norm with p > 0.0 is only supported");

  auto output =
      at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  auto retain =
      at::empty(self.sizes(), self.options(), self.suggest_memory_format());

  ns_LpNormKernel::Params params{};
  params.p = p.to<float>();
  params.dim = 0;
  params.eps = 1e-5;

  std::vector<at::Tensor> outputs{output, retain};
  AllocateSynapseOutputs(graph, outputs, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
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
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  PT_KERNEL_END;
  return out.at(0);
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::native_batch_norm",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<BatchNormForwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::native_batch_norm_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<BatchNormBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::native_layer_norm",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<LayerNormOperator>(device_id, node_type);
            })
        .add(
            "aten::native_layer_norm_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<LayerNormBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::norm.scalar",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<NormOperator>(device_id, node_type);
            });
