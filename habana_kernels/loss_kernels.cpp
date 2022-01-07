/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/core/Reduction.h>
#include <perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/loss_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/threshold_kernels.h"
#include "habana_kernels/unary_kernels.h"
#include "simple_generic_kernel.h"
#include "synapse_helpers/recipe.h"

using namespace torch;
using namespace habana;

static ns_NLLLossKernel::ParamsOptionalIgnoreIndex
synapse_nll_loss_params_builder(int64_t reduction, int64_t ignore_index) {
  auto param = ns_NLLLossKernel::ParamsOptionalIgnoreIndex{};
  if (reduction == at::Reduction::Reduction::None) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_NONE;
  } else if (reduction == at::Reduction::Reduction::Mean) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_MEAN;
  } else if (reduction == at::Reduction::Reduction::Sum) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_SUM;
  } else
    TORCH_CHECK(false, "nll_loss got unsuported reduction type: ", reduction);
  param.ignoreIndexValue = (int)ignore_index;

  return param;
}

static ns_BinaryCrossEntropy::ParamsOptionalSigmoid synapse_bce_params_builder(
    int64_t reduction,
    bool weightsDefined) {
  auto param = ns_BinaryCrossEntropy::ParamsOptionalSigmoid{};
  if (reduction == at::Reduction::Reduction::Mean) {
    param.mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_MEAN;
  } else if (reduction == at::Reduction::Reduction::Sum) {
    param.mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_SUM;
  } else
    TORCH_CHECK(
        false, "BinaryCrossEntropy got unsuported reduction type: ", reduction);
  param.binaryCrossEntropyWithoutSigmoid = true;
  param.isWeightsUsed = weightsDefined;
  return param;
}

static ns_BinaryCrossEntropy::ParamsOptionalPosWeight
synapse_bce_logits_params_builder(
    int64_t reduction,
    bool posWeightsDefined,
    bool weightsDefined) {
  auto param = ns_BinaryCrossEntropy::ParamsOptionalPosWeight{};
  if (reduction == at::Reduction::Reduction::Mean) {
    param.mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_MEAN;
  } else if (reduction == at::Reduction::Reduction::Sum) {
    param.mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_SUM;
  } else {
    HABANA_ASSERT(0 && "https://jira.habana-labs.com/browse/SW-36304")
    param.mode = ECrossEntropyMode_t::CROSS_ENTROPY_MODE_NO_REDUCTION;
  }
  param.posMode = posWeightsDefined ? POS_WEIGHT_ENABLE : POS_WEIGHT_DISABLE;
  param.isWeightsUsed = weightsDefined;
  return param;
}

static ns_MSELossKernel::Params synapse_mse_loss_params_builder(
    int64_t reduction) {
  auto param = ns_MSELossKernel::Params{};
  if (reduction == at::Reduction::Reduction::None) {
    param.mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_NONE;
  } else if (reduction == at::Reduction::Reduction::Mean) {
    param.mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_MEAN;
  } else if (reduction == at::Reduction::Reduction::Sum) {
    param.mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_SUM;
  } else
    TORCH_CHECK(false, "mse_loss got unsuported reduction type: ", reduction);

  return param;
}

void NLLLossFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for nll_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 expected to be tensor");
  TORCH_CHECK(
      inputs[2].isTensor() || inputs[2].isNone(),
      "Input arg3 expected to be tensor or None for nll_loss operator");
  TORCH_CHECK(
      inputs[3].isInt(),
      "Input arg 3 expected to be of type Int for nll_loss operator");
  TORCH_CHECK(
      inputs[4].isInt(),
      "Input arg 3 expected to be of type Int for nll_loss operator");

  // Should assert right here if we are asked to handle weights
  if (inputs[2].isTensor()) {
    TORCH_CHECK(
        !inputs[2].toTensor().defined(),
        "NLL loss kernel does not support weights for now");
  } else {
    TORCH_CHECK(
        inputs[2].isNone(), "NLL kernel does not support weights for now");
  }

  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  int64_t reduction = inputs[3].toInt();
  int64_t ignore_index = inputs[4].toInt();
  TORCH_CHECK(
      target.scalar_type() == c10::ScalarType::Int,
      "Input arg 2 expected to be of Int Tensor for nll_loss operator");
  ns_NLLLossKernel::ParamsOptionalIgnoreIndex params =
      synapse_nll_loss_params_builder(reduction, ignore_index);
  p_context_->params_.emplace<ns_NLLLossKernel::ParamsOptionalIgnoreIndex>(
      params);
  p_context_->params_size_ = sizeof(params);
  std::vector<int64_t> reshaped_self_sizes;
  if (reduction == at::Reduction::Reduction::None) {
    reshaped_self_sizes.emplace_back(self.sizes()[0]);
  } else {
    reshaped_self_sizes.emplace_back(1);
  }
  auto output1 = habana_helpers::createPTTensor(
      self,
      reshaped_self_sizes,
      self.options(),
      at::MemoryFormat::Contiguous,
      is_output_persistent[0]);
  AllocateSynapseOutput(graph, output1, is_output_persistent[0]);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));

  // create a dummy output, we do not support weights therefore there is no
  // sum_weights tensor, but we still need to return an empty tensor to keep
  // Pytorch happy
  auto output2 = habana_helpers::createPTTensor(
      self,
      {1},
      self.options(),
      at::MemoryFormat::Contiguous,
      is_output_persistent[1]);
  p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
      output2, graph, is_output_persistent[1], c10::nullopt));
  p_context_->pt_outputs_.emplace_back(output2);
}

void NLLLoss2dFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for nll_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 expected to be tensor");
  TORCH_CHECK(
      inputs[2].isTensor() || inputs[2].isNone(),
      "Input arg3 expected to be tensor or None for nll_loss operator");
  TORCH_CHECK(
      inputs[3].isInt(),
      "Input arg 3 expected to be of type Int for nll_loss operator");
  TORCH_CHECK(
      inputs[4].isInt(),
      "Input arg 3 expected to be of type Int for nll_loss operator");

  // Should assert right here if we are asked to handle weights
  if (inputs[2].isTensor()) {
    TORCH_CHECK(
        !inputs[2].toTensor().defined(),
        "NLL loss kernel does not support weights for now");
  } else {
    TORCH_CHECK(
        inputs[2].isNone(), "NLL kernel does not support weights for now");
  }

  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  int64_t reduction = inputs[3].toInt();
  int64_t ignore_index = inputs[4].toInt();

  TORCH_CHECK(
      target.scalar_type() == c10::ScalarType::Int,
      "Input arg 2 expected to be of Int Tensor for nll_loss operator");
  ns_NLLLossKernel::ParamsOptionalIgnoreIndex params =
      synapse_nll_loss_params_builder(reduction, ignore_index);
  p_context_->params_.emplace<ns_NLLLossKernel::ParamsOptionalIgnoreIndex>(
      params);
  p_context_->params_size_ = sizeof(params);

  std::vector<int64_t> reshaped_self_sizes;
  if (reduction == at::Reduction::Reduction::None) {
    reshaped_self_sizes.emplace_back(self.sizes()[0]);
    reshaped_self_sizes.emplace_back(self.sizes()[1]);
    reshaped_self_sizes.emplace_back(self.sizes()[2]);
  } else {
    reshaped_self_sizes.emplace_back(1);
  }
  auto output1 = habana_helpers::createPTTensor(
      self,
      reshaped_self_sizes,
      self.options(),
      c10::MemoryFormat::Contiguous,
      is_output_persistent[0]);
  AllocateSynapseOutput(graph, output1, is_output_persistent[0]);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  // create a dummy output, we do not support weights therefore there is no
  // sum_weights tensor, but we still need to return an empty tensor to keep
  // Pytorch happy
  auto output2 = habana_helpers::createPTTensor(
      self,
      {1},
      self.options(),
      at::MemoryFormat::Contiguous,
      is_output_persistent[1]);
  p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
      output2, graph, is_output_persistent[1], c10::nullopt));
  p_context_->pt_outputs_.emplace_back(output2);
}

/** @brief Function implements forward pass for torch.nn.NLLLoss
 *  @param self: Input tensor of shape (N,C), where C = Number of classes.
 *  @param target: Input tensor of shape (N), where each value 0 <= i < C.
 *  @param weight: (Tensor, Optional) a manual rescaling weight given to each
 * class. If given, it has to be a Tensor of size C. Otherwise, it is treated as
 * if having all ones.
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 *  @param ignore_index: (Long, Optional) Specifies a target value that is
 * Fix me :ignore_index is not supported in this implementation
 * ignored and does not contribute to the input gradient.
 */
std::tuple<Tensor, Tensor> nll_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")

  auto modified_target = habana_helpers::cast_tensor_to_integer(target);

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "nll_loss_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self),
      IValue(modified_target),
      IValue(weight),
      IValue(reduction),
      IValue(ignore_index)};
  NLLLossFwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self, modified_target};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    std::vector<int64_t> reshaped_self_sizes;
    if (reduction == at::Reduction::Reduction::None) {
      reshaped_self_sizes.emplace_back(self.sizes()[0]);
    } else {
      reshaped_self_sizes.emplace_back(1);
    }
    auto output1 = at::empty(
        reshaped_self_sizes, self.options(), at::MemoryFormat::Contiguous);
    auto output2 = at::empty({1}, self.options(), at::MemoryFormat::Contiguous);
    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{output1, output2};
    Op.SetPTOutputs(v);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  if (reduction != at::Reduction::Reduction::None) {
    // Note: pytorch expects 0d tensor (scalar)
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  PT_KERNEL_END;
  return std::make_tuple(out.at(0), out.at(1));
}

std::tuple<Tensor, Tensor> nll_loss2d_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")

  Tensor self_nhwc = self;
  if (self.ndimension() == 4) {
    // convert tensors to synapse memory format
    int64_t pos_in[] = {0, 2, 3, 1};
    std::vector<const at::Tensor*> pt_in{&self};
    std::vector<at::Tensor*> pt_out{&self_nhwc};
    IntArrayRef new_dim_pos_in = pos_in;
    std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in};
    c10::MemoryFormat memory_format =
        habana_helpers::get_memory_format({&self});
    habana_helpers::change_tensors_to_memory_format(
        pt_out, pt_in, pt_new_pos, memory_format);
  }

  auto modified_target = habana_helpers::cast_tensor_to_integer(target);
  at::ScalarType scalar_type = self_nhwc.scalar_type();
  std::string node_type =
      "nll_loss_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self_nhwc.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self_nhwc),
      IValue(modified_target),
      IValue(weight),
      IValue(reduction),
      IValue(ignore_index)};
  NLLLoss2dFwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self_nhwc, modified_target};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    std::vector<int64_t> reshaped_self_sizes;
    if (reduction == at::Reduction::Reduction::None) {
      reshaped_self_sizes.emplace_back(self.sizes()[0]);
      reshaped_self_sizes.emplace_back(self.sizes()[1]);
      reshaped_self_sizes.emplace_back(self.sizes()[2]);
    } else {
      reshaped_self_sizes.emplace_back(1);
    }
    auto output0 = at::empty(
        reshaped_self_sizes,
        self_nhwc.options(),
        self_nhwc.suggest_memory_format());
    auto output1 =
        at::empty({1}, self_nhwc.options(), self_nhwc.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{output0, output1};
    Op.SetPTOutputs(v);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  if (reduction != at::Reduction::Reduction::None) {
    // Note: pytorch expects 0d tensor (scalar)
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  PT_KERNEL_END;
  return std::make_tuple(out.at(0), out.at(1));
}

void NLLLossBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs expected for nll_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(
      inputs[3].isTensor() || inputs[3].isNone(),
      "Input type expected to be tensor or None");
  TORCH_CHECK(inputs[4].isInt(), "Input type expected to be Int");
  TORCH_CHECK(inputs[5].isInt(), "Input type expected to be Int");
  TORCH_CHECK(
      inputs[6].isTensor() || inputs[6].isNone(),
      "Input type expected to be tensor or None");

  // Should assert right here if we are asked to handle weights
  if (inputs[3].isTensor()) {
    TORCH_CHECK(
        !inputs[3].toTensor().defined(),
        "NLL Loss kernel does not support weights for now");
  } else {
    TORCH_CHECK(
        inputs[3].isNone(), "NLL Loss kernel does not support weights for now");
  }

  auto self = inputs[1].toTensor();
  int64_t reduction = inputs[4].toInt();
  int64_t ignore_index = inputs[5].toInt();

  ns_NLLLossKernel::ParamsOptionalIgnoreIndex params =
      synapse_nll_loss_params_builder(reduction, ignore_index);
  p_context_->params_.emplace<ns_NLLLossKernel::ParamsOptionalIgnoreIndex>(
      params);
  p_context_->params_size_ = sizeof(params);

  auto output = habana_helpers::createPTTensor(self, is_output_persistent);
  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void NLLLoss2dBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs expected for nll_loss2d operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(
      inputs[3].isTensor() || inputs[3].isNone(),
      "Input type expected to be tensor or None");
  TORCH_CHECK(inputs[4].isInt(), "Input type expected to be Int");
  TORCH_CHECK(inputs[5].isInt(), "Input type expected to be Int");
  TORCH_CHECK(
      inputs[6].isTensor() || inputs[6].isNone(),
      "Input type expected to be tensor or None");

  // Should assert right here if we are asked to handle weights
  if (inputs[3].isTensor()) {
    TORCH_CHECK(
        !inputs[3].toTensor().defined(),
        "NLL Loss kernel does not support weights for now");
  } else {
    TORCH_CHECK(
        inputs[3].isNone(), "NLL Loss kernel does not support weights for now");
  }

  auto self = inputs[1].toTensor();
  int64_t reduction = inputs[4].toInt();
  int64_t ignore_index = inputs[5].toInt();

  ns_NLLLossKernel::ParamsOptionalIgnoreIndex params =
      synapse_nll_loss_params_builder(reduction, ignore_index);
  p_context_->params_.emplace<ns_NLLLossKernel::ParamsOptionalIgnoreIndex>(
      params);
  p_context_->params_size_ = sizeof(params);

  auto output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      c10::MemoryFormat::Contiguous,
      is_output_persistent);

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, output);
  }

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/** @brief Function implements backward pass for torch.nn.NLLLoss
 *  @param grad_output: Input (bwd_pass) tensor of shape N or 1.
 *  @param self: Input (fwd_pass) tensor of shape (N,C), where C = Number of
 * classes.
 *  @param target: Input tensor (fwd_pass) of shape (N), where each value 0 <= i
 * < C.
 *  @param weight: (Tensor, Optional) a manual rescaling weight given to each
 * class. If given, it has to be a Tensor of size C. Otherwise, it is treated as
 * if having all ones.
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 *  @param ignore_index: (Long, Optional) Specifies a target value that is
 * ignored and does not contribute to the input gradient.
 * Fix me :ignore_index is not supported in this implementation
 *  @param total_weight: (single element tensor) sum of weights used in fwd_pass
 */
Tensor nll_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (total_weight.dim() == 0) {
    total_weight.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto modified_target = habana_helpers::cast_tensor_to_integer(target);

  at::ScalarType scalar_type = grad_output.scalar_type();
  std::string node_type =
      "nll_loss_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = grad_output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack = {
      IValue(grad_output),
      IValue(self),
      IValue(target),
      IValue(weight),
      IValue(reduction),
      IValue(ignore_index),
      IValue(total_weight)};
  NLLLossBwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{
      grad_output, self, modified_target, total_weight};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

Tensor nll_loss2d_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")

  Tensor self_nhwc = self;
  if (self.ndimension() == 4) {
    // convert tensors to synapse memory format
    int64_t pos_in[] = {0, 2, 3, 1};
    std::vector<const at::Tensor*> pt_in{&self};
    std::vector<at::Tensor*> pt_out{&self_nhwc};
    IntArrayRef new_dim_pos_in = pos_in;
    std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in};
    c10::MemoryFormat memory_format =
        habana_helpers::get_memory_format({&self});
    habana_helpers::change_tensors_to_memory_format(
        pt_out, pt_in, pt_new_pos, memory_format);
  }

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (total_weight.dim() == 0) {
    total_weight.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto modified_target = habana_helpers::cast_tensor_to_integer(target);

  at::ScalarType scalar_type = grad_output.scalar_type();
  std::string node_type =
      "nll_loss_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = grad_output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<c10::IValue> stack = {
      IValue(grad_output),
      IValue(self_nhwc),
      IValue(target),
      IValue(weight),
      IValue(reduction),
      IValue(ignore_index),
      IValue(total_weight)};
  NLLLoss2dBwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{
      grad_output, self_nhwc, modified_target, total_weight};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        self_nhwc.sizes(), self_nhwc.options(), c10::MemoryFormat::Contiguous);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (self.ndimension() == 4) {
    // convert tensors to synapse memory format
    Tensor result = out.at(0);
    Tensor result_nchw = out.at(0);
    int64_t pos_in[] = {0, 3, 1, 2};
    std::vector<const at::Tensor*> pt_in{&result};
    std::vector<at::Tensor*> pt_out{&result_nchw};
    IntArrayRef new_dim_pos_in = pos_in;
    std::vector<const IntArrayRef*> pt_new_pos{&new_dim_pos_in};
    c10::MemoryFormat memory_format =
        habana_helpers::get_memory_format({&result});
    habana_helpers::change_tensors_to_memory_format(
        pt_out, pt_in, pt_new_pos, memory_format);

    PT_KERNEL_END;
    return result_nchw;
  }
  PT_KERNEL_END;
  return out.at(0);
}

std::vector<int64_t> MSELossFwdOperator::compute_output_shape(
    const at::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return self.sizes().vec();
  } else {
    return {};
  }
}

void MSELossFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for mse_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  int64_t reduction = inputs[2].toInt();

  ns_MSELossKernel::Params param = synapse_mse_loss_params_builder(reduction);
  p_context_->params_.emplace<ns_MSELossKernel::Params>(param);
  p_context_->params_size_ = sizeof(param);

  auto sizes = MSELossFwdOperator::compute_output_shape(self, reduction);
  auto output = habana_helpers::createPTTensor(
      self,
      sizes,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

/** @brief Function implements forward pass for torch.nn.MSELoss
 *  @param self: Input tensor of shape (N,C)
 *  @param target: Input tensor of shape (N,C)
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 */
Tensor mse_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "mse_loss_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue{target}, IValue(reduction)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, target};

  MSELossFwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Tensor output;
    if (reduction == at::Reduction::Reduction::None) {
      output = habana_helpers::createPTTensor(self, true);
    } else {
      output = habana_helpers::createPTTensor(
          self, {}, self.options(), self.suggest_memory_format(), true);
    }
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();

  if (reduction != at::Reduction::Reduction::None) {
    // Note: pytorch expects 0d tensor (scalar)
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  PT_KERNEL_END;
  return out.at(0);
}

std::vector<int64_t> MSELossBwdOperator::compute_output_shape(
    const Tensor& self) {
  return self.sizes().vec();
}

void MSELossBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for mse_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");

  auto self = inputs[1].toTensor();
  int64_t reduction = inputs[3].toInt();

  ns_MSELossKernel::Params param = synapse_mse_loss_params_builder(reduction);
  p_context_->params_.emplace<ns_MSELossKernel::Params>(param);
  p_context_->params_size_ = sizeof(param);

  auto output = habana_helpers::createPTTensor(self, is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

/** @brief Function implements backward pass for torch.nn.MSELoss
 *  @param grad_output: Input (bwd_pass) tensor of shape (N,C) or 1.
 *  @param self: Input (fwd_pass) tensor of shape (N,C)
 *  @param target: Input tensor (fwd_pass) of shape (N,C)
 *  @param reduction: (String, Optional) Specifies the reduction to apply to the
 * output: 'none' | 'mean' | 'sum'.
 */
Tensor mse_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_KERNEL_BEGIN;

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "mse_loss_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue{grad_output}, IValue{self}, IValue{target}, IValue{reduction}};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad_output, self, target};

  MSELossBwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

std::vector<int64_t> KlDivOperator::compute_output_shape(
    const at::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return self.sizes().vec();
  } else {
    return {};
  }
}

void KlDivOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for kl_div operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isInt(), "Input type expected to be integer");
  TORCH_CHECK(inputs[3].isBool(), "Input type expected to be boolean");

  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  int64_t reduction = inputs[2].toInt();
  bool log_target = inputs[3].toBool();

  torch::jit::Stack stack;
  HabanaOperatorPtr log_exp_op;
  HabanaOperatorPtr threshold_op;
  if (log_target) {
    log_exp_op = static_cast<HabanaOperatorPtr>(
        make_operator<ExpOperator>(self.device().index(), self.scalar_type()));
    stack = {IValue(target)};
    log_exp_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    log_exp_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

  } else {
    log_exp_op = static_cast<HabanaOperatorPtr>(
        make_operator<LogOperator>(self.device().index(), self.scalar_type()));
    stack = {IValue(target)};
    log_exp_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    log_exp_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    threshold_op = make_operator<ThresholdBackwardOperator>(
        self.device().index(), self.scalar_type());
    threshold_op->SetSynapseInput(log_exp_op->GetSynOutputs()[0]);
    threshold_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    stack = {IValue(log_exp_op->GetOutputs()[0]), IValue(target), IValue(0.0f)};
    threshold_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();
  }

  auto sub_op =
      make_operator<SubOperator>(self.device().index(), self.scalar_type());
  (log_target) ? sub_op->SetSynapseInput(p_context_->syn_inputs_[1])
               : sub_op->SetSynapseInput(threshold_op->GetSynOutputs()[0]);
  sub_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  stack = {
      (log_target) ? IValue(target) : IValue(threshold_op->GetOutputs()[0]),
      IValue(self),
      IValue(1)};
  sub_op->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  auto mul_op1 =
      make_operator<MulOperator>(self.device().index(), self.scalar_type());
  (log_target) ? mul_op1->SetSynapseInput(log_exp_op->GetSynOutputs()[0])
               : mul_op1->SetSynapseInput(p_context_->syn_inputs_[1]);
  mul_op1->SetSynapseInput(sub_op->GetSynOutputs()[0]);
  if (reduction == at::Reduction::Reduction::None) {
    mul_op1->SetOutputMetadata(output_metadata_);
  }
  stack = {
      (log_target) ? IValue(log_exp_op->GetOutputs()[0]) : IValue(target),
      IValue(sub_op->GetOutputs()[0])};
  mul_op1->AllocateAndAddSynapseNode(
      graph,
      stack,
      (reduction == at::Reduction::Reduction::None) ? is_output_persistent
                                                    : false);
  stack.clear();

  if (reduction != at::Reduction::Reduction::None) {
    auto sum_mean_op = (reduction == at::Reduction::Reduction::Sum)
        ? static_cast<HabanaOperatorPtr>(make_operator<SumOperator>(
              self.device().index(), self.scalar_type()))
        : static_cast<HabanaOperatorPtr>(make_operator<MeanOperator>(
              self.device().index(), self.scalar_type()));
    sum_mean_op->SetSynapseInput(mul_op1->GetSynOutputs()[0]);
    sum_mean_op->SetOutputMetadata(output_metadata_);
    stack = {IValue(mul_op1->GetOutputs()[0]), IValue(self.scalar_type())};
    sum_mean_op->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(sum_mean_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(sum_mean_op->GetOutputs()[0]));
  } else {
    p_context_->syn_outputs_.emplace_back(
        std::move(mul_op1->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mul_op1->GetOutputs()[0]));
  }
}

Tensor kl_div_hpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "kl_div_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue{target}, IValue(reduction), IValue(log_target)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, target};

  KlDivOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Tensor output;
    if (reduction == at::Reduction::Reduction::None) {
      output = habana_helpers::createPTTensor(self, true);
    } else {
      output = habana_helpers::createPTTensor(
          self, {}, self.options(), self.suggest_memory_format(), true);
    }
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (reduction != at::Reduction::Reduction::None) {
    // Note: pytorch expects 0d tensor (scalar)
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  PT_KERNEL_END;
  return out.at(0);
}
void KlDivBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for kl_div backward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isInt(), "Input type expected to be integer");
  TORCH_CHECK(inputs[4].isBool(), "Input type expected to be boolean");

  auto grad_out = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto target = inputs[2].toTensor();
  int64_t reduction = inputs[3].toInt();
  bool log_target = inputs[4].toBool();

  torch::jit::Stack stack;
  HabanaOperatorPtr exp_op;
  if (log_target) {
    exp_op = static_cast<HabanaOperatorPtr>(
        make_operator<ExpOperator>(self.device().index(), self.scalar_type()));
    stack = {IValue(target)};
    exp_op->SetSynapseInput(p_context_->syn_inputs_[2]);
    exp_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();
  }

  if (reduction != at::Reduction::Reduction::None) {
    auto shape = self.sizes().vec();
    int flattened_size = std::accumulate(
        shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
    int64_t data[1];
    data[0] = flattened_size;
    IntArrayRef size_arr(data, 1);
    auto sum_mean_op = (reduction == at::Reduction::Reduction::Sum)
        ? static_cast<HabanaOperatorPtr>(make_operator<ReduceSumBwdOperator>(
              self.device().index(), self.scalar_type()))
        : static_cast<HabanaOperatorPtr>(make_operator<ReduceMeanBwdOperator>(
              self.device().index(), self.scalar_type()));
    sum_mean_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    stack = {IValue(grad_out), IValue(size_arr), IValue(0)};
    sum_mean_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto shape1 = self.sizes().vec();
    auto reshape_op = make_operator<ReshapeOperator>(
        self.device().index(), self.scalar_type());
    reshape_op->SetSynapseInput(sum_mean_op->GetSynOutputs()[0]);
    stack = {IValue(sum_mean_op->GetOutputs()[0]), IValue(shape1)};
    reshape_op->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto mul_op1 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op1->SetSynapseInput(reshape_op->GetSynOutputs()[0]);
    (log_target) ? mul_op1->SetSynapseInput(exp_op->GetSynOutputs()[0])
                 : mul_op1->SetSynapseInput(p_context_->syn_inputs_[2]);
    stack = {
        IValue(reshape_op->GetOutputs()[0]),
        (log_target) ? IValue(exp_op->GetOutputs()[0]) : IValue(target)};
    mul_op1->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto mul_op3 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op3->SetSynapseInput(mul_op1->GetSynOutputs()[0]);
    mul_op3->SetOutputMetadata(output_metadata_);
    stack = {IValue(mul_op1->GetOutputs()[0]), IValue(-1)};
    mul_op3->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(mul_op3->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mul_op3->GetOutputs()[0]));
  } else {
    auto mul_op2 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op2->SetSynapseInput(p_context_->syn_inputs_[0]);
    (log_target) ? mul_op2->SetSynapseInput(exp_op->GetSynOutputs()[0])
                 : mul_op2->SetSynapseInput(p_context_->syn_inputs_[2]);
    stack = {
        IValue(grad_out),
        (log_target) ? IValue(exp_op->GetOutputs()[0]) : IValue(target)};
    mul_op2->AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    auto mul_op4 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op4->SetSynapseInput(mul_op2->GetSynOutputs()[0]);
    mul_op4->SetOutputMetadata(output_metadata_);
    stack = {IValue(mul_op2->GetOutputs()[0]), IValue(-1)};
    mul_op4->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(mul_op4->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mul_op4->GetOutputs()[0]));
  }
}

Tensor kl_div_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    bool log_target) {
  PT_KERNEL_BEGIN;

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "kl_div_backward_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue{grad_output},
      IValue{self},
      IValue{target},
      IValue{reduction},
      IValue{log_target}};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad_output, target};

  KlDivBwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

std::vector<int64_t> BceFwdOperator::compute_output_shape(
    const at::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return self.sizes().vec();
  } else {
    return {};
  }
}

void BceFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4, "Incorrect size of inputs expected for BCE operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for BCE operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for BCE operator");
  TORCH_CHECK(
      inputs[2].isTensor() || inputs[2].isNone(),
      "Input arg3 expected to be tensor or None for BCE operator");
  TORCH_CHECK(
      inputs[3].isInt(),
      "Input arg 4 expected to be of type Int for BCE operator");

  if (inputs[2].isTensor()) {
    TORCH_CHECK(
        !inputs[2].toTensor().defined(),
        "BCE kernel does not support weights for now");
  } else {
    TORCH_CHECK(
        inputs[2].isNone(), "BCE kernel does not support weights for now");
  }

  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  int64_t reduction = inputs[3].toInt();

  // add reshape node to reverse input dims
  auto reshape_self =
      make_operator<ReshapeOperator>(self.device().index(), self.scalar_type());
  reshape_self->SetSynapseInput(p_context_->syn_inputs_[0]);
  auto v = self.sizes().vec();
  std::reverse(std::begin(v), std::end(v));
  torch::jit::Stack stack = {IValue(self), IValue(v)};
  reshape_self->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // add reshape node to make target same shape as reshaped input
  auto reshape_target = make_operator<ReshapeOperator>(
      target.device().index(), target.scalar_type());
  reshape_target->SetSynapseInput(p_context_->syn_inputs_[1]);
  stack = {IValue(target), IValue(v)};
  reshape_target->AllocateAndAddSynapseNode(graph, stack, false);

  // fill params for BCE node
  ns_BinaryCrossEntropy::ParamsOptionalSigmoid params =
      synapse_bce_params_builder(reduction, false);
  p_context_->params_.emplace<ns_BinaryCrossEntropy::ParamsOptionalSigmoid>(
      params);
  p_context_->params_size_ = sizeof(params);

  // set-up input/output tensors for BCE
  auto sizes = BceFwdOperator::compute_output_shape(self, reduction);
  auto output = habana_helpers::createPTTensor(
      self, sizes, self.options(), is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  synapse_helpers::tensor& syn_in_self = reshape_self->GetSynOutputs()[0];
  synapse_helpers::tensor& syn_in_tensor = reshape_target->GetSynOutputs()[0];
  std::vector<synTensor> syn_inputs{syn_in_self.get(), syn_in_tensor.get()};
  synapse_helpers::tensor& syn_out = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{syn_out.get()};

  // add BCE node
  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      std::move(guid_));
}

Tensor binary_cross_entropy_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type = "binary_cross_entropy_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(target), IValue(weight), IValue(reduction)};
  BceFwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self, target};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty({self.sizes()[1]}, self.options());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  // Note: pytorch expects 0d tensor (scalar)
  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  PT_KERNEL_END;
  return out.at(0);
}

void BceBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5, "Incorrect size of inputs expected for BCE operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for BCE operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for BCE operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for BCE operator");
  TORCH_CHECK(
      inputs[3].isTensor() || inputs[3].isNone(),
      "Input arg4 expected to be tensor or None for BCE operator");
  TORCH_CHECK(
      inputs[4].isInt(),
      "Input arg 5 expected to be of type Int for BCE operator");

  if (inputs[3].isTensor()) {
    TORCH_CHECK(
        !inputs[3].toTensor().defined(),
        "BCE kernel does not support weights for now");
  } else {
    TORCH_CHECK(
        inputs[3].isNone(), "BCE kernel does not support weights for now");
  }

  auto grad_output = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto target = inputs[2].toTensor();
  int64_t reduction = inputs[4].toInt();

  // add reshape node to reverse input dims
  auto reshape_self =
      make_operator<ReshapeOperator>(self.device().index(), self.scalar_type());
  reshape_self->SetSynapseInput(p_context_->syn_inputs_[1]);
  auto v = self.sizes().vec();
  std::reverse(std::begin(v), std::end(v));
  torch::jit::Stack stack = {IValue(self), IValue(v)};
  reshape_self->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // add reshape node to make target same shape as reshaped input
  auto reshape_target = make_operator<ReshapeOperator>(
      target.device().index(), target.scalar_type());
  reshape_target->SetSynapseInput(p_context_->syn_inputs_[2]);
  stack = {IValue(target), IValue(v)};
  reshape_target->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  auto neg_grad = make_operator<NegOperator>(
      grad_output.device().index(), grad_output.scalar_type());
  neg_grad->SetSynapseInput(p_context_->syn_inputs_[0]);
  stack = {IValue(grad_output)};
  neg_grad->AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  ns_BinaryCrossEntropy::ParamsOptionalSigmoid params =
      synapse_bce_params_builder(reduction, false);
  p_context_->params_.emplace<ns_BinaryCrossEntropy::ParamsOptionalSigmoid>(
      params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(
      graph,
      habana_helpers::createPTTensor(reshape_self->GetOutputs()[0], false),
      false,
      false,
      false);
  synapse_helpers::tensor& syn_in_self = reshape_self->GetSynOutputs()[0];
  synapse_helpers::tensor& syn_in_target = reshape_target->GetSynOutputs()[0];
  synapse_helpers::tensor& syn_in_grad = neg_grad->GetSynOutputs()[0];
  std::vector<synTensor> syn_inputs{
      syn_in_self.get(), syn_in_target.get(), syn_in_grad.get()};
  synapse_helpers::tensor& syn_out = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{syn_out.get()};

  // add BCE node
  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      &params,
      sizeof(params),
      std::move(guid_));

  // add reshape node on output
  auto reshape_grad_in =
      make_operator<ReshapeOperator>(self.device().index(), self.scalar_type());
  reshape_grad_in->SetSynapseInput(p_context_->syn_outputs_[0]);
  stack = {
      c10::IValue(p_context_->pt_outputs_[0]), c10::IValue(self.sizes().vec())};
  reshape_grad_in->SetOutputMetadata(output_metadata_);
  reshape_grad_in->AllocateAndAddSynapseNode(
      graph, stack, is_output_persistent);
  stack.clear();

  p_context_->syn_outputs_[0] = std::move(reshape_grad_in->GetSynOutputs()[0]);
  p_context_->pt_outputs_[0] = reshape_grad_in->GetOutputs()[0];
}

Tensor binary_cross_entropy_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  PT_KERNEL_BEGIN;

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  at::ScalarType scalar_type = grad_output.scalar_type();
  std::string node_type;
  node_type = "binary_cross_entropy_bwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = grad_output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(grad_output),
      IValue(self),
      IValue(target),
      IValue(weight),
      IValue(reduction)};
  BceBwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{grad_output, self, target};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty_like(self);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  auto output = out.at(0);

  PT_KERNEL_END;
  return output;
}

std::vector<int64_t> BceLogitsFwdOperator::compute_output_shape(
    const at::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return self.sizes().vec();
  } else {
    return {};
  }
}

void BceLogitsFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for BCELogits operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for BCELogits operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for BCELogits operator");
  TORCH_CHECK(
      inputs[2].isTensor() || inputs[2].isNone(),
      "Input arg3 expected to be tensor or None for BCELogits operator");
  TORCH_CHECK(
      inputs[3].isTensor() || inputs[3].isNone(),
      "Input arg4 expected to be tensor or None for BCELogits operator");
  TORCH_CHECK(
      inputs[4].isInt(),
      "Input arg5 expected to be of type Int for BCELogits operator");

  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  auto pos_weight = inputs[2].isTensor() ? inputs[2].toTensor()
                                         : inputs[2].toOptional<Tensor>();
  auto weight = inputs[3].isTensor() ? inputs[3].toTensor()
                                     : inputs[3].toOptional<Tensor>();
  auto reduction = inputs[4].toInt();

  ns_BinaryCrossEntropy::ParamsOptionalPosWeight param =
      synapse_bce_logits_params_builder(
          reduction, inputs[2].isTensor(), inputs[3].isTensor());
  p_context_->params_.emplace<ns_BinaryCrossEntropy::ParamsOptionalPosWeight>(
      param);
  p_context_->params_size_ = sizeof(param);

  Tensor output;
  auto sizes = BceLogitsFwdOperator::compute_output_shape(self, reduction);
  output = habana_helpers::createPTTensor(
      self, sizes, self.options(), is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

Tensor binary_cross_entropy_with_logits_hpu(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& pos_weight,
    int64_t reduction) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type = "binary_cross_entropy_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self),
      IValue(target),
      IValue(weight),
      IValue(pos_weight),
      IValue(reduction)};
  BceLogitsFwdOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self, target};
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Tensor output;
    if (reduction == at::Reduction::Reduction::Mean ||
        reduction == at::Reduction::Reduction::Sum) {
      output = habana_helpers::createPTTensor(self, {}, self.options(), true);
    } else {
      output = habana_helpers::createPTTensor(
          self, self.sizes(), self.options(), true);
    }
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  // Note: pytorch expects 0d tensor (scalar)
  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  PT_KERNEL_END;
  return out.at(0);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("aten::binary_cross_entropy", KERNEL_FN(BceFwdOperator))
        .add("aten::binary_cross_entropy_backward", KERNEL_FN(BceBwdOperator))
        .add(
            "aten::binary_cross_entropy_with_logits",
            KERNEL_FN(BceLogitsFwdOperator))
        .add("aten::mse_loss", KERNEL_FN(MSELossFwdOperator))
        .add("aten::mse_loss_backward", KERNEL_FN(MSELossBwdOperator))
        .add("aten::kl_div", KERNEL_FN(KlDivOperator))
        .add("aten::kl_div_backward", KERNEL_FN(KlDivBwdOperator))
        .add("aten::nll_loss_forward", KERNEL_FN(NLLLossFwdOperator))
        .add("aten::nll_loss_backward", KERNEL_FN(NLLLossBwdOperator))
        .add("aten::nll_loss2d_forward", KERNEL_FN(NLLLoss2dFwdOperator))
        .add("aten::nll_loss2d_backward", KERNEL_FN(NLLLoss2dBwdOperator));
