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
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/unary_kernels.h"
using namespace torch;
using namespace habana;

bool is_5d_tensor(const std::vector<int64_t>& shape_in) {
  const uint64_t DIM5 = 5;
  return shape_in.size() == DIM5;
}

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

void BatchNormForwardOperator::insert_memcopy_op(
    synapse_helpers::graph& graph,
    at::Tensor& src,
    int in_position) {
  auto memcopyOp = make_operator<MemCopyOperator>(
      this->p_context_->device_id_, this->scalarType_);
  auto& syn_temp = memcopyOp->SetSynapseInput(
      std::move(p_context_->syn_inputs_[in_position]));
  // No need for output PT tensor as its non persistent
  torch::jit::Stack stack = {IValue(src)};
  memcopyOp->AllocateAndAddSynapseNode(graph, stack, false);
  synapse_helpers::tensor& syn_tensor = memcopyOp->GetSynOutputs()[0];
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
    return ret_tensor;
  } else {
    return input;
  }
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
    auto syn_tensor =
        habana_helpers::create_tensor(ret_tensor, graph, true, c10::nullopt);
    auto it = p_context_->syn_inputs_.begin() + syn_index;
    p_context_->syn_inputs_.insert(it, std::move(syn_tensor));

    appended_tensor_infos.emplace_back(
        std::make_tuple(syn_tensor.name(), ret_tensor, syn_tensor.id()));
    return ret_tensor;
  } else {
    return input;
  }
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
  auto device = DeviceType::HPU;

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
    pre_inputs = {
        std::move(input),
        std::move(wt_hpu),
        std::move(bias_hpu),
        std::move(residualAdd),
        std::move(running_mean_hpu),
        std::move(running_var_hpu)};
    // Mean and var only passed once as intermediate ones are non-persistent
    // adn dont go for patching
    pt_inputs = {
        pre_inputs[0],
        pre_inputs[1],
        pre_inputs[2],
        pre_inputs[3],
        pre_inputs[4],
        pre_inputs[5]};
    input_stack = {
        IValue(pre_inputs[0]),
        IValue(pre_inputs[4]),
        IValue(pre_inputs[5]),
        IValue(training),
        IValue(momentum),
        IValue(eps)};
  } else {
    // training=false - Evaluation mode
    pre_inputs = {
        std::move(input),
        std::move(bias_hpu),
        std::move(wt_hpu),
        std::move(running_mean_hpu),
        std::move(running_var_hpu)};
    pt_inputs = {
        pre_inputs[0],
        pre_inputs[1],
        pre_inputs[2],
        pre_inputs[3],
        pre_inputs[4]};
    input_stack = {
        IValue(pre_inputs[0]),
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
  auto device = DeviceType::HPU;

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
      insert_memcopy_op(graph, running_mean_hpu, 3);
      insert_memcopy_op(graph, running_var_hpu, 4);
    }
    // Mean and Var syn tensors may be reused as they are not bound to data
    // create a new syn tensor for bias and add it
    // This residual add is dummy tensor to match the API requirements
    residualAdd = at::empty(input.sizes(), input.options());
    auto syn_tensor_add =
        habana_helpers::create_tensor(residualAdd, graph, true, c10::nullopt);
    auto it = p_context_->syn_inputs_.begin() + 3;

    // This is to communicate to graph lowering that a new tensor was
    // added by the kernel and it can add to patching in lowering
    appended_tensor_infos.emplace_back(std::make_tuple(
        syn_tensor_add.name(), residualAdd, syn_tensor_add.id()));
    p_context_->syn_inputs_.insert(it, std::move(syn_tensor_add));

    pre_inputs = {
        std::move(input),
        std::move(wt_hpu),
        std::move(bias_hpu),
        std::move(residualAdd),
        std::move(running_mean_hpu),
        std::move(running_var_hpu)};
    pt_inputs = {
        pre_inputs[0],
        pre_inputs[1],
        pre_inputs[2],
        pre_inputs[3],
        pre_inputs[4],
        pre_inputs[5],
        pre_inputs[4],
        pre_inputs[5]};
    input_stack = {
        IValue(pre_inputs[0]),
        IValue(pre_inputs[4]),
        IValue(pre_inputs[5]),
        IValue(training),
        IValue(momentum),
        IValue(eps)};
  } else {
    // training=false - Evaluation mode
    pre_inputs = {
        std::move(input),
        std::move(bias_hpu),
        std::move(wt_hpu),
        std::move(running_mean_hpu),
        std::move(running_var_hpu)};
    pt_inputs = {
        pre_inputs[0],
        pre_inputs[1],
        pre_inputs[2],
        pre_inputs[3],
        pre_inputs[4]};
    input_stack = {
        IValue(pre_inputs[0]),
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
    struct synCudBnExParams params = {
        synBnOps::BN_OPS_BN,
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
          mean_var_temp[0], graph);
      auto syn_tensor_var = habana_helpers::duplicate_tensor_in_memory_section(
          mean_var_temp[1], graph);

      appended_tensor_infos.emplace_back(std::make_tuple(
          syn_tensor_mean.name(), pre_inputs[4], syn_tensor_mean.id()));
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_mean));

      appended_tensor_infos.emplace_back(std::make_tuple(
          syn_tensor_var.name(), pre_inputs[5], syn_tensor_var.id()));
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_var));
    } else {
      // As the tensors are used as IO and are persistent
      // WE need to use same mem section
      auto syn_tensor_mean = habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[3], graph);
      auto syn_tensor_var = habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[4], graph);

      appended_tensor_infos.emplace_back(std::make_tuple(
          syn_tensor_mean.name(), pre_inputs[4], syn_tensor_mean.id()));
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_mean));

      appended_tensor_infos.emplace_back(std::make_tuple(
          syn_tensor_var.name(), pre_inputs[5], syn_tensor_var.id()));
      p_context_->syn_outputs_.emplace_back(std::move(syn_tensor_var));
    }

    p_context_->pt_outputs_.emplace_back(pre_inputs[4]);
    p_context_->pt_outputs_.emplace_back(pre_inputs[5]);

    std::vector<bool> persistent_output_flags{
        is_output_persistent[1], is_output_persistent[2]};
    AllocateSynapseOutputs(
        graph,
        {current_mean, current_istd},
        persistent_output_flags,
        {true, true});

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
  if (isEagerMode()) {
    remove_non_persistent_patching_info();
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
    std::vector<at::Tensor> v{
        output, running_mean_hpu, running_var_hpu, current_mean, current_istd};
    HabanaOperator::SetPTOutputs(v);
  } else {
    std::vector<at::Tensor> v{output, running_mean_hpu, running_var_hpu};
    HabanaOperator::SetPTOutputs(v);
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
  Stack cache_stack = {
      IValue(input),
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
  Stack in_stack = {
      IValue(input_nhwc),
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
      Op.SetEagerMode();
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      for (auto ival : in_stack) {
        if (ival.isTensor() && ival.toTensor().defined()) {
          at::Tensor in = ival.toTensor().to(DeviceType::HPU);
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

/*********************************************************************************
 * @brief - This op is used in lazy mode to avoid memcopy nodes for RMV
 *********************************************************************************/
at::Tensor BatchNormForwardRmvOperator::create_or_return_tensor_bn(
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
    auto syn_tensor =
        habana_helpers::create_tensor(ret_tensor, graph, true, c10::nullopt);
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
at::Tensor BatchNormForwardRmvOperator::create_or_return_pt_tensor_bn(
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

void BatchNormForwardRmvOperator::preProcessInputs(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect number of inputs against expected count for BatchNormForward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[6].isBool(), "Input type expected to be bool");
  TORCH_CHECK(inputs[7].isDouble(), "Input type expected to be double");
  TORCH_CHECK(inputs[8].isDouble(), "Input type expected to be double");

  const auto input = inputs[0].toTensor();
  const auto weight = inputs[1].toTensor();
  const auto bias = inputs[2].toTensor();
  const auto residual_add = inputs[3].toTensor();
  const auto running_mean = inputs[4].toTensor();
  const auto running_var = inputs[5].toTensor();
  const auto training = inputs[6].toBool();

  std::string guid = training ? "cud_bn_fwd_ex"
                              : "batch_norm_inf_" +
          habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  Tensor wt_hpu, bias_hpu;
  auto device = DeviceType::HPU;

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

  Tensor running_mean_hpu = create_or_return_tensor_bn(
      graph, running_mean, input.sizes()[3], device, (uint)4);
  Tensor running_var_hpu = create_or_return_tensor_bn(
      graph, running_var, input.sizes()[3], device, (uint)5);

  pre_inputs = {
      std::move(input),
      std::move(wt_hpu),
      std::move(bias_hpu),
      std::move(residual_add),
      std::move(running_mean_hpu),
      std::move(running_var_hpu)};
  pt_inputs = {
      pre_inputs[0],
      pre_inputs[1],
      pre_inputs[2],
      pre_inputs[3],
      pre_inputs[4],
      pre_inputs[5]};

  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
}

void BatchNormForwardRmvOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    std::vector<bool> is_output_persistent) {
  // Add intermediate tensors and add to op
  preProcessInputs(graph, in_stack);
  TORCH_CHECK(in_stack[6].isBool(), "Input type expected to be bool");
  TORCH_CHECK(in_stack[7].isDouble(), "Input type expected to be double");
  TORCH_CHECK(in_stack[8].isDouble(), "Input type expected to be double");
  TORCH_CHECK(
      is_output_persistent.size() == 5,
      "BatchNormForwardOperator: is_output_persistent should be 5 in training mode");
  const auto momentum = in_stack[7].toDouble();
  const auto eps = in_stack[8].toDouble();

  auto output =
      habana_helpers::createPTTensor(pre_inputs[0], is_output_persistent[0]);

  AllocateSynapseOutput(graph, output, is_output_persistent[0]);
  auto current_mean =
      habana_helpers::createPTTensor(pre_inputs[4], is_output_persistent[1]);
  auto current_istd =
      habana_helpers::createPTTensor(pre_inputs[5], is_output_persistent[2]);
  std::vector<bool> persistent_output_flags{
      is_output_persistent[1], is_output_persistent[2]};
  AllocateSynapseOutputs(
      graph,
      {current_mean, current_istd},
      persistent_output_flags,
      {true, true});
  auto running_mean_out =
      habana_helpers::createPTTensor(pre_inputs[4], is_output_persistent[3]);
  auto running_var_out =
      habana_helpers::createPTTensor(pre_inputs[5], is_output_persistent[4]);

  AllocateSynapseOutputs(
      graph,
      {running_mean_out, running_var_out},
      {is_output_persistent[3], is_output_persistent[4]},
      {true, true});

  // synapse uses expAvgfactor = 1 - momentum
  struct synCudBnExParams params = {
      synBnOps::BN_OPS_BN,
      static_cast<float>(1 - momentum),
      static_cast<float>(eps)};
  p_context_->params_.emplace<synCudBnExParams>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void BatchNormInfOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& in_stack,
    bool is_output_persistent) {
  TORCH_CHECK(in_stack[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(in_stack[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(in_stack[5].isBool(), "Input type expected to be bool");
  TORCH_CHECK(in_stack[6].isDouble(), "Input type expected to be double");
  TORCH_CHECK(in_stack[7].isDouble(), "Input type expected to be double");

  auto input = in_stack[0].toTensor();
  const auto momentum = in_stack[6].toDouble();
  const auto eps = in_stack[7].toDouble();

  std::string guid = "batch_norm_inf_" +
      habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  auto output = habana_helpers::createPTTensor(input, is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);

  struct ns_BatchNormKernel::Params params;
  params.threshold.f = 0.0;
  params.momentum = static_cast<float>(momentum);
  params.epsilon = static_cast<float>(eps);
  p_context_->params_.emplace<ns_BatchNormKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void BatchNormBackwardOperator::create_opt_input_tensor_bn_bwd(
    synapse_helpers::graph& graph,
    const Tensor& input,
    uint size,
    Device device,
    int pos) {
  Tensor ret_tensor;
  if (!input.defined()) {
    ret_tensor = at::empty({size}, device);
    auto syn_tensor =
        habana_helpers::create_tensor(ret_tensor, graph, true, c10::nullopt);
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
      inputs.size() == 7,
      "Incorrect number of inputs against expected count for BatchNormBackward preProcessInputs: expected 7 got ",
      inputs.size());
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");

  const auto input = inputs[1].toTensor();
  const auto weight = inputs[2].toTensor();
  const auto running_mean = inputs[3].toTensor();
  const auto running_var = inputs[4].toTensor();
  const auto save_mean = inputs[5].toTensor();
  const auto save_invstd = inputs[6].toTensor();

  Tensor bias_hpu;

  auto device = input.device();

  create_opt_input_tensor_bn_bwd(graph, weight, input.sizes()[3], device, 2);
  create_opt_input_tensor_bn_bwd(
      graph, running_mean, input.sizes()[3], device, 3);
  create_opt_input_tensor_bn_bwd(
      graph, running_var, input.sizes()[3], device, 4);
  create_opt_input_tensor_bn_bwd(graph, save_mean, input.sizes()[3], device, 5);
  create_opt_input_tensor_bn_bwd(
      graph, save_invstd, input.sizes()[3], device, 6);
  create_opt_input_tensor_bn_bwd(graph, bias_hpu, input.sizes()[3], device, 7);

  SetProprocessingDone();
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
  const auto input = inputs[1].toTensor();
  const auto weight = inputs[2].toTensor();
  const auto eps = inputs[8].toDouble();

  // Prepare output tensor vector
  auto grad_in_nhwc =
      habana_helpers::createPTTensor(input, is_output_persistent[0]);
  auto grad_beta =
      habana_helpers::createPTTensor(weight, is_output_persistent[2]);
  auto grad_gamma =
      habana_helpers::createPTTensor(weight, is_output_persistent[1]);

  struct synCudBnExParams params = {
      synBnOps::BN_OPS_BN, 0, static_cast<float>(eps)};
  p_context_->params_.emplace<synCudBnExParams>(params);
  p_context_->params_size_ = sizeof(params);
  // TODO: swap metadata order to match allocation order?
  AllocateSynapseOutputs(
      graph,
      {grad_in_nhwc, grad_gamma, grad_beta},
      is_output_persistent,
      {true, true, true});
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
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

  auto device = input.device();

  auto opt_tensor = at::empty({input.sizes()[3]}, device);

  std::vector<at::Tensor> pt_inputs{
      grad_out,
      input,
      weight.defined() ? weight : opt_tensor,
      running_mean.defined() ? running_mean : opt_tensor,
      running_var.defined() ? running_var : opt_tensor,
      save_mean.defined() ? save_mean : opt_tensor,
      save_invstd.defined() ? save_invstd : opt_tensor,
      opt_tensor /* bias */
  };

  p_context_->pt_inputs_.clear();
  SetPTInputs(pt_inputs);
  input_stack = {
      IValue(pt_inputs[1]),
      IValue(pt_inputs[2])}; // for creating output tensors
  SetProprocessingDone();
}

void BatchNormBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  const auto input = inputs[0].toTensor();
  const auto wt_hpu = inputs[1].toTensor();
  // Prepare output tensor vector
  auto grad_in_nhwc = at::empty(input.sizes(), input.options());
  auto grad_beta = at::empty(wt_hpu.sizes(), wt_hpu.options());
  auto grad_gamma = at::empty(wt_hpu.sizes(), wt_hpu.options());
  std::vector<at::Tensor> v{grad_in_nhwc, grad_gamma, grad_beta};
  HabanaOperator::SetPTOutputs(v);
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
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> output_mask) {
  PT_KERNEL_BEGIN;
  // Build Params for the graph
  Stack cache_stack = {
      IValue(grad_out),
      IValue(input),
      IValue(weight),
      IValue(running_mean),
      IValue(running_var),
      IValue(save_mean),
      IValue(save_invstd),
      IValue(train),
      IValue(eps),
      IValue(output_mask.data())};
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
  Stack preprocess_stack = {
      IValue(grad_out_nhwc),
      IValue(input_nhwc),
      IValue(weight),
      IValue(running_mean),
      IValue(running_var),
      IValue(save_mean),
      IValue(save_invstd),
      IValue(train),
      IValue(eps),
      IValue(output_mask.data())};
  auto batch_norm_bwd = [&] {
    // Create the operator
    BatchNormBackwardOperator Op(device_id, scalar_type);

    Op.SetResizeDone(); // true for eager mode
    size_t key = Op.GetRecipeKey(node_type, cache_stack);
    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      Stack cache_preprocess_stack = {
          IValue(grad_out_nhwc),
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
          at::Tensor in = ival.toTensor().to(DeviceType::HPU);
          Op.AllocateSynapseInput(graph, in, true);
        }
      }
      // Build Params for the graph
      Op.AllocateAndAddSynapseNode(graph, preprocess_stack, {true, true, true});
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

std::vector<std::vector<int64_t>> LayerNormOperator::getOutputSizes(
    const at::Tensor& input,
    int m) {
  auto output_sizes = input.sizes().vec();
  std::vector<int64_t> shape_mean{1, 1, m, 1};
  return std::vector<std::vector<int64_t>>{
      output_sizes, shape_mean, shape_mean};
}
std::tuple<Tensor, Tensor, Tensor> LayerNormOperator::AllocatePTOutputs(
    const Tensor& input,
    const Tensor& bias,
    const Tensor& weight,
    int64_t m,
    std::array<bool, 3> is_persistent) {
  std::vector<int64_t> shape_mean{1, 1, m, 1};
  auto sizes = LayerNormOperator::getOutputSizes(input, m);
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
      inputs.size() == 5,
      "LayerNormOperator::AllocateAndAddSynapseNode expected 5 args but got ",
      inputs.size())

  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isIntList(), "Input type expected to be int list");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isDouble(), "Input type expected to be doube");

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
  reshape_op_input->AllocateAndAddSynapseNode(graph, stack, false);
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
  reshape_op_bias->AllocateAndAddSynapseNode(graph, stack, false);
  auto bias_reshaped = reshape_op_bias->GetOutputs()[0];
  synapse_helpers::tensor& syn_bias_ln = reshape_op_bias->GetSynOutputs()[0];

  // Add Reshape node for weight to graph for weight.view(-1)
  auto reshape_op_wt = make_operator<ReshapeOperator>(
      weight.device().index(), weight.scalar_type());
  reshape_op_wt->SetSynapseInput(p_context_->syn_inputs_[1]);
  sizes[0] = weight.numel();
  c10::IntArrayRef modified_weight_shape(sizes, 1);
  stack = {c10::IValue(weight), c10::IValue(modified_weight_shape)};
  reshape_op_wt->AllocateAndAddSynapseNode(graph, stack, false);
  auto wt_reshaped = reshape_op_wt->GetOutputs()[0];
  synapse_helpers::tensor& syn_wt_ln = reshape_op_wt->GetSynOutputs()[0];

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

  std::vector<OutputMetaData> out_0_metadata =
      SelectVectorIndices(output_metadata_, {0});
  output_metadata_.erase(output_metadata_.begin());
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
  auto reshape_op_out = make_operator<ReshapeOperator>(
      input.device().index(), input.scalar_type());
  reshape_op_out->SetSynapseInput(p_context_->syn_outputs_[0]);
  stack = {c10::IValue(input_reshaped), c10::IValue(input.sizes().vec())};
  reshape_op_out->SetOutputMetadata(out_0_metadata);
  reshape_op_out->AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[0]);
  synapse_helpers::tensor& syn_reshape_out = reshape_op_out->GetSynOutputs()[0];
  p_context_->syn_outputs_[0] = std::move(syn_reshape_out);
  p_context_->pt_outputs_[0] = reshape_op_out->GetOutputs()[0];
}

void LayerNormOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 5,
      "LayerNormOperator::AllocateAndAddSynapseNode expected 5 args but got ",
      inputs.size())
  const auto input = inputs[0].toTensor();
  const auto normalized_shape = inputs[1].toIntList().vec();
  const auto weight = inputs[2].toTensor();
  const auto bias = inputs[3].toTensor();

  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  const int axis = input_ndim - normalized_ndim;
  int64_t m =
      multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);

  auto outputs = AllocatePTOutputs(input, bias, weight, m, {true, true, true});
  std::vector<at::Tensor> v{
      std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)};
  HabanaOperator::SetPTOutputs(v);
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
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    double eps) {
  PT_KERNEL_BEGIN;
  // Build Params for the graph
  auto weight = weight_opt.value();
  auto bias = bias_opt.value();
  Stack input_stack = {
      IValue(input),
      IValue(normalized_shape),
      IValue(weight),
      IValue(bias),
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
      const std::vector<at::Tensor> pt_inputs{input, weight, bias};
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

// This was added for Lazy Mode and is being used only in Lazy mode unit
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
      inputs.size());
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isIntList(), "Input type expected to be int list");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[6].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[7].isBoolList(), "Input type expected to be bool array");

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
  reshape_op_x->AllocateAndAddSynapseNode(graph, stack, false);
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
  reshape_op_dy->AllocateAndAddSynapseNode(graph, stack, false);
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
  reshape_op_gamma->AllocateAndAddSynapseNode(graph, stack, false);
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
  reshape_op_mean->AllocateAndAddSynapseNode(graph, stack, false);
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
  reshape_op_rstd->AllocateAndAddSynapseNode(graph, stack, false);
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
      false,
      false,
      false);
  AllocateSynapseOutput(
      graph,
      habana_helpers::createPTTensor(gamma_reshaped, false),
      false,
      false,
      false);
  AllocateSynapseOutput(
      graph,
      habana_helpers::createPTTensor(gamma_reshaped, false),
      false,
      false,
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
      std::move(guid_));

  // Add Reshape nodes for output tensors to graph
  auto reshape_op_grad_in =
      make_operator<ReshapeOperator>(dY.device().index(), dY.scalar_type());
  reshape_op_grad_in->SetSynapseInput(p_context_->syn_outputs_[0]);
  reshape_op_grad_in->SetOutputMetadata(
      SelectVectorIndices(output_metadata_, {0}));
  stack = {c10::IValue(output0), c10::IValue(output0.sizes().vec())};
  reshape_op_grad_in->AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[0]);
  synapse_helpers::tensor& syn_reshape_grad_in =
      reshape_op_grad_in->GetSynOutputs()[0];

  auto reshape_op_grad_gamma = make_operator<ReshapeOperator>(
      gamma.device().index(), gamma.scalar_type());
  reshape_op_grad_gamma->SetSynapseInput(p_context_->syn_outputs_[2]);
  reshape_op_grad_gamma->SetOutputMetadata(
      SelectVectorIndices(output_metadata_, {1}));
  stack = {c10::IValue(output2), c10::IValue(output2.sizes().vec())};
  reshape_op_grad_gamma->AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[1]);
  synapse_helpers::tensor& syn_reshape_grad_gamma =
      reshape_op_grad_gamma->GetSynOutputs()[0];

  auto reshape_op_grad_beta = make_operator<ReshapeOperator>(
      gamma.device().index(), gamma.scalar_type());
  reshape_op_grad_beta->SetSynapseInput(p_context_->syn_outputs_[1]);
  reshape_op_grad_beta->SetOutputMetadata(
      SelectVectorIndices(output_metadata_, {2}));
  stack = {c10::IValue(output1), c10::IValue(output1.sizes().vec())};
  reshape_op_grad_beta->AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent[2]);
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

void LayerNormBackwardOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  const auto dY = inputs[0].toTensor();
  const auto gamma = inputs[5].toTensor();

  auto outputs = AllocatePTOutputs(dY, gamma, true);
  std::vector<at::Tensor> v{
      std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs)};
  HabanaOperator::SetPTOutputs(v);
}

std::vector<std::vector<int64_t>> LayerNormBackwardOperator::getOutputSizes(
    const at::Tensor& input,
    const at::Tensor& gamma) {
  std::vector<int64_t> gamma_size =
      gamma.defined() ? gamma.sizes().vec() : input.sizes().vec();
  return std::vector<std::vector<int64_t>>{
      input.sizes().vec(), gamma_size, gamma_size};
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
    const at::Tensor& dY,
    const at::Tensor& X,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> grad_input_mask) {
  PT_KERNEL_BEGIN;
  std::vector<bool> grad_mask_in;
  grad_mask_in.push_back(grad_input_mask[0]);
  grad_mask_in.push_back(grad_input_mask[1]);
  grad_mask_in.push_back(grad_input_mask[2]);

  // Build Params for the graph
  Stack input_stack = {
      IValue(dY),
      IValue(X),
      IValue(normalized_shape),
      IValue(mean),
      IValue(rstd),
      IValue(weight_opt),
      IValue(bias_opt),
      IValue(grad_mask_in)};
  size_t device_id = X.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = X.scalar_type();
  std::string node_type = "layer_norm";
  std::vector<at::Tensor> out;
  auto gamma = weight_opt.value();
  auto bias = bias_opt.value();
  auto layer_norm = [&] {
    // Create the operator
    LayerNormBackwardOperator Op(device_id, scalar_type);

    size_t key = Op.GetRecipeKey(node_type, input_stack);

    if (device.get_recipe_handle_cache().isCached(key)) {
      PT_KERNEL_DEBUG("Cache hit key:", key);
      // Assign Inputs to the Operator
      const std::vector<at::Tensor> pt_inputs{dY, X, mean, rstd, gamma, bias};
      Op.SetPTInputs(pt_inputs);
      Op.SetPTOutputs(input_stack);
      Op.Execute(key);
      out = Op.GetOutputs();
    } else {
      // Create Graph
      auto graph = habana_helpers::create_graph(device_id, node_type);
      const std::vector<at::Tensor> pt_inputs{dY, X, mean, rstd, gamma, bias};
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

std::vector<int64_t> NormOperator::compute_output_shape() {
  return {1};
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
  auto shape = NormOperator::compute_output_shape();
  auto output = at::empty(shape, self.options(), c10::nullopt);
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

  if (p.toFloat() == 2.0) {
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
      identityOp->AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      auto mulOp = make_operator<habana::MulOperator>(
          this->p_context_->device_id_, scalar_type);
      mulOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      mulOp->SetSynapseInput(identityOp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(self));
      stack.emplace_back(IValue(identityOp->GetOutputs()[0]));
      mulOp->AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();
      // add node to compute reduce_sum
      auto sum_lp = make_operator<SumOperator>(device_id, scalar_type);
      sum_lp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
      stack.emplace_back(IValue(scalar_type));
      sum_lp->AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      auto sqrt_op = make_operator<SqrtOperator>(device_id, scalar_type);
      sqrt_op->SetSynapseInput(sum_lp->GetSynOutputs()[0]);
      sqrt_op->SetOutputMetadata(output_metadata_);
      stack.emplace_back(IValue(sum_lp->GetOutputs()[0]));
      sqrt_op->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
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
      LpNormFrobeniusOp->SetOutputMetadata(output_metadata_);

      // Build Params for the graph
      stack.emplace_back(IValue(self));
      LpNormFrobeniusOp->AllocateAndAddSynapseNode(
          graph, stack, is_output_persistent);

      stack.clear();
      p_context_->syn_outputs_.emplace_back(
          std::move(LpNormFrobeniusOp->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(LpNormFrobeniusOp->GetOutputs()[0]);
    }
  } else {
    // ReShape Operator
    at::ScalarType scalar_type = self.scalar_type();
    auto shape = {self.numel()};

    // Create the operator
    auto ReShapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, scalar_type);
    ReShapeOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::IntArrayRef(shape))};
    ReShapeOp->AllocateAndAddSynapseNode(graph, stack, false);

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
    LpNormOp->AllocateAndAddSynapseNode(graph, stack, {false, false});

    auto output_norm = LpNormOp->GetOutputs()[1];
    stack.clear();

    // Reciprocal Operator
    // Create the operator
    auto reciprocalOp = make_operator<ReciprocalOperator>(
        this->p_context_->device_id_, scalar_type);
    reciprocalOp->SetSynapseInput(LpNormOp->GetSynOutputs()[1]);

    // Build Params for the graph
    stack.emplace_back(IValue(output_norm));
    reciprocalOp->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // take just the first element of Reciprocal since all values would be
    // repeated
    auto slice_op =
        make_operator<SliceOperator>(this->p_context_->device_id_, scalar_type);
    slice_op->SetSynapseInput(reciprocalOp->GetSynOutputs()[0]);
    slice_op->SetOutputMetadata(output_metadata_);
    stack.emplace_back(IValue(reciprocalOp->GetOutputs()[0]));
    int dim = 0;
    int start = 0;
    int end = 1;
    int step = 1;
    stack.emplace_back(IValue(dim));
    stack.emplace_back(IValue(start));
    stack.emplace_back(IValue(end));
    stack.emplace_back(IValue(step));
    slice_op->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    p_context_->syn_outputs_.emplace_back(
        std::move(slice_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(slice_op->GetOutputs()[0]));
  }
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

  auto output = habana_helpers::createPTTensor(self, is_output_persistent[0]);
  auto retain = habana_helpers::createPTTensor(self, is_output_persistent[1]);

  ns_LpNormKernel::Params params{};
  params.p = p.to<float>();
  params.dim = 0;
  params.eps = 1e-5;

  std::vector<at::Tensor> outputs{output, retain};
  AllocateSynapseOutputs(graph, outputs, is_output_persistent, {true, true});
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void LpNormFrobeniusOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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
      is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);
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

std::shared_ptr<SliceOperator> FusedNormOperator::compute_clip_coeff(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool>& is_output_persistent) {
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
      norm_lp->AllocateAndAddSynapseNode(graph, stack, false);
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
    cat_grad_norms->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // node to do compute total_norm
    auto norm_final = make_operator<NormOperator>(device_id, scalar_type);
    norm_final->SetSynapseInput(cat_grad_norms->GetSynOutputs()[0]);
    norm_final->SetOutputMetadata(SelectVectorIndices(output_metadata_, {0}));
    stack.emplace_back(IValue(cat_grad_norms->GetOutputs()[0]));
    stack.emplace_back(IValue(2.0));
    norm_final->AllocateAndAddSynapseNode(
        graph, stack, is_output_persistent[0]);
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
    add_op1->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // clip_coef = max_norm / (total_norm + 1e-6)
    auto div_final = make_operator<DivOperator>(device_id, scalar_type);
    div_final->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    div_final->SetSynapseInput(add_op1->GetSynOutputs()[0]);
    stack.emplace_back(IValue(max_grad_norm));
    stack.emplace_back(IValue(add_op1->GetOutputs()[0]));
    div_final->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // mask = total_norm > max_grad_norm
    auto gt_op = make_operator<GtOperator>(device_id, scalar_type);
    gt_op->SetSynapseInput(p_context_->syn_outputs_[0]);
    gt_op->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    stack.emplace_back(IValue(p_context_->pt_outputs_[0]));
    stack.emplace_back(IValue(max_grad_norm));
    gt_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    std::string node_type = "cast_i8_to_f32";
    auto cast1 = make_operator<CastOperator>(device_id, node_type);
    cast1->SetSynapseInput(gt_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(gt_op->GetOutputs()[0]));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    cast1->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // mul1 = mask * clip_coef
    auto mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(cast1->GetSynOutputs()[0]);
    mul1->SetSynapseInput(div_final->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(div_final->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();
    // imask = (mask == 0)
    auto eq_op = make_operator<EqOperator>(device_id, scalar_type);
    eq_op->SetSynapseInput(cast1->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(0.0));
    eq_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    node_type = "cast_i8_to_f32";
    auto cast2 = make_operator<CastOperator>(device_id, node_type);
    cast2->SetSynapseInput(eq_op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(eq_op->GetOutputs()[0]));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    cast2->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // mask*clip_coef + imask
    auto add_op2 = make_operator<AddOperator>(device_id, scalar_type);
    add_op2->SetSynapseInput(mul1->GetSynOutputs()[0]);
    add_op2->SetSynapseInput(cast2->GetSynOutputs()[0]);
    stack.emplace_back(IValue(mul1->GetOutputs()[0]));
    stack.emplace_back(IValue(cast2->GetOutputs()[0]));
    stack.emplace_back(IValue(1.0));
    add_op2->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();
    // take just the first element of add since all values would be repeated

    slice_op->SetSynapseInput(add_op2->GetSynOutputs()[0]);
    stack.emplace_back(IValue(add_op2->GetOutputs()[0]));
    stack.emplace_back(IValue(0));
    stack.emplace_back(IValue(0));
    stack.emplace_back(IValue(1));
    stack.emplace_back(IValue(1));
    slice_op->AllocateAndAddSynapseNode(graph, stack, false);
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
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for FusedNorm Operator");

  auto gradients = inputs[0].toTensorList();
  auto num_params = static_cast<unsigned int>(gradients.size());
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  auto slice_op = compute_clip_coeff(graph, inputs, is_output_persistent);

  torch::jit::Stack stack;

  // p.grad.detach().mul_(clip_coef)
  for (unsigned int i = 0; i < num_params; i++) {
    auto mul1 = make_operator<MulInplaceOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(p_context_->syn_inputs_[i]);
    mul1->SetSynapseInput(slice_op->GetSynOutputs()[0]);
    mul1->SetOutputMetadata(SelectVectorIndices(output_metadata_, {i + 1u}));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(slice_op->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(graph, stack, is_output_persistent[i + 1]);
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
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for FusedNorm Operator");

  auto gradients = inputs[0].toTensorList();
  auto num_params = static_cast<unsigned int>(gradients.size());
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  auto slice_op = compute_clip_coeff(graph, inputs, is_output_persistent);

  torch::jit::Stack stack;

  // clipped_grad = p.grad.detach().mul(clip_coef)
  for (unsigned int i = 0; i < num_params; i++) {
    auto mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(p_context_->syn_inputs_[i]);
    mul1->SetSynapseInput(slice_op->GetSynOutputs()[0]);
    mul1->SetOutputMetadata(SelectVectorIndices(output_metadata_, {i + 1u}));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(slice_op->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(graph, stack, is_output_persistent[i + 1]);
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
    std::vector<bool> is_output_persistent(num_params + 1, true);
    Op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(
      out.size() == ((unsigned)num_params + 1), "Incorrect size of outputs");
  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});

  PT_KERNEL_END;
  return out[0];
}

std::vector<int64_t> InstanceNormOperator::compute_output_shape(
    at::Tensor input,
    c10::MemoryFormat mf) {
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
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(in_stack[3].isDouble(), "Input type expected to be double");
  TORCH_CHECK(
      is_output_persistent.size() == 3,
      "InstanceNormOperator: is_output_persistent should be 3 in training mode");

  auto input = in_stack[0].toTensor();

  auto is_norm_3d = is_5d_tensor(input.sizes().vec());

  std::string guid = "instance_norm_fwd_" +
      habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  auto beta = in_stack[1].toTensor();

  const auto eps = in_stack[3].toDouble();

  auto output = habana_helpers::createPTTensor(input, is_output_persistent[0]);
  AllocateSynapseOutput(graph, output, is_output_persistent[0]);

  auto memory_format = is_norm_3d ? c10::MemoryFormat::ChannelsLast3d
                                  : c10::MemoryFormat::ChannelsLast;
  auto mean_var_shape =
      InstanceNormOperator::compute_output_shape(input, memory_format);

  auto current_mean = habana_helpers::createPTTensor(
      beta,
      mean_var_shape,
      beta.options(),
      c10::MemoryFormat::Contiguous,
      is_output_persistent[1]);

  auto current_istd = habana_helpers::createPTTensor(
      beta,
      mean_var_shape,
      beta.options(),
      c10::MemoryFormat::Contiguous,
      is_output_persistent[2]);

  std::vector<bool> persistent_output_flags{
      is_output_persistent[1], is_output_persistent[2]};
  AllocateSynapseOutputs(
      graph,
      {current_mean, current_istd},
      persistent_output_flags,
      {true, true});

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
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      is_output_persistent.size() == 3,
      "InstanceNormOperator: is_output_persistent should be 3 in training mode");
  auto input = in_stack[0].toTensor();
  auto mean = in_stack[2].toTensor();

  auto is_norm_3d = is_5d_tensor(input.sizes().vec());

  std::string guid = "instance_norm_bwd_" +
      habana_helpers::name_suffix_from_type(input.scalar_type());
  SetGuid(guid);

  auto output = habana_helpers::createPTTensor(input, is_output_persistent[0]);
  AllocateSynapseOutput(graph, output, is_output_persistent[0]);

  auto memory_format = is_norm_3d ? c10::MemoryFormat::ChannelsLast3d
                                  : c10::MemoryFormat::ChannelsLast;
  auto grad_beta_gamma_shape =
      InstanceNormBackwardOperator::compute_output_shape(input, memory_format);

  auto grad_beta = habana_helpers::createPTTensor(
      mean,
      grad_beta_gamma_shape,
      mean.options(),
      c10::MemoryFormat::Contiguous,
      is_output_persistent[1]);

  auto grad_gamma = habana_helpers::createPTTensor(
      mean,
      grad_beta_gamma_shape,
      mean.options(),
      c10::MemoryFormat::Contiguous,
      is_output_persistent[2]);

  std::vector<bool> persistent_output_flags{
      is_output_persistent[1], is_output_persistent[2]};
  AllocateSynapseOutputs(
      graph, {grad_beta, grad_gamma}, persistent_output_flags, {true, true});

  // Note: TPC kernel doesnt support running mean and variance computation. we
  // just pass random momentum value as a place holder
  struct ns_InstanceNormTrainingKernel::Params params {
    0.9, 1e-5
  };

  p_context_->params_.emplace<ns_InstanceNormTrainingKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::native_batch_norm", KERNEL_FN(BatchNormForwardOperator))
        .add(
            "hpu::native_batch_norm_rmv",
            KERNEL_FN(BatchNormForwardRmvOperator))
        .add("hpu::native_batch_norm_inf", KERNEL_FN(BatchNormInfOperator))
        .add("hpu::fused_norm_", KERNEL_FN(FusedNormOperator))
        .add("hpu::fused_norm_lazy", KERNEL_FN(FusedNormLazyOperator))
        .add(
            "aten::native_batch_norm_backward",
            KERNEL_FN(BatchNormBackwardOperator))
        .add("aten::native_layer_norm", KERNEL_FN(LayerNormOperator))
        .add(
            "aten::native_layer_norm_backward",
            KERNEL_FN(LayerNormBackwardOperator))
        .add("aten::norm.Scalar", KERNEL_FN(NormOperator))
        .add("hpu::instance_norm", KERNEL_FN(InstanceNormOperator))
        .add(
            "hpu::instance_norm_backward",
            KERNEL_FN(InstanceNormBackwardOperator));
