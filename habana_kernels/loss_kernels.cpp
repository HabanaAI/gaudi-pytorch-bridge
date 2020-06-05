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
#include "habana_kernels/loss_kernels.h"
#include "simple_generic_kernel.h"

using namespace torch;
using namespace habana;

static ns_NLLLossKernel::Params synapse_nll_loss_params_builder(
    int64_t reduction) {
  auto param = ns_NLLLossKernel::Params{};
  if (reduction == at::Reduction::Reduction::None) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_NONE;
  } else if (reduction == at::Reduction::Reduction::Mean) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_MEAN;
  } else if (reduction == at::Reduction::Reduction::Sum) {
    param.mode = NLLLossMode_t::NLL_LOSS_MODE_SUM;
  } else
    TORCH_CHECK(false, "nll_loss got unsuported reduction type: ", reduction);

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
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for nll_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 expected to be tensor");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input arg 3 expected to be of type Int for nll_loss operator");

  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  int64_t reduction = inputs[2].toInt();

  TORCH_CHECK(
      target.scalar_type() == c10::ScalarType::Int,
      "Input arg 2 expected to be of Int Tensor for nll_loss operator");

  ns_NLLLossKernel::Params param = synapse_nll_loss_params_builder(reduction);
  p_context_->params_.emplace<ns_NLLLossKernel::Params>(param);
  p_context_->params_size_ = sizeof(param);

  auto output = at::empty({1}, self.options());
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
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
 * ignored and does not contribute to the input gradient.
 */
std::tuple<Tensor, Tensor> nll_loss_forward_hpu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")
  TORCH_CHECK(ignore_index == -100, "ignore_index is not yet supported")

  auto modified_target = habana_helpers::cast_tensor_to_integer(target);

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "nll_loss_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  NLLLossFwdOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&self, &modified_target};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(modified_target), IValue(reduction)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  // Note: pytorch expects 0d tensor (scalar)
  out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  LOG_FUNC_END;
  return std::make_tuple(out.at(0), at::empty({0}, self.options()));
}

void NLLLossBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for nll_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  int64_t reduction = inputs[1].toInt();

  ns_NLLLossKernel::Params param = synapse_nll_loss_params_builder(reduction);
  p_context_->params_.emplace<ns_NLLLossKernel::Params>(param);
  p_context_->params_size_ = sizeof(param);

  auto output = at::empty(self.sizes(), self.options());
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
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
 *  @param total_weight: (single element tensor) sum of weights used in fwd_pass
 */
Tensor nll_loss_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  LOG_FUNC_BEGIN;
  TORCH_CHECK(!weight.defined(), "weighted nll_loss is not yet supported")
  TORCH_CHECK(ignore_index == -100, "ignore_index is not yet supported")

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto modified_target = habana_helpers::cast_tensor_to_integer(target);

  at::ScalarType scalar_type = grad_output.scalar_type();
  std::string node_type =
      "nll_loss_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = grad_output.device().index();

  NLLLossBwdOperator Op(device_id, node_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&grad_output, &modified_target};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(reduction)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  LOG_FUNC_END;
  return out.at(0);
}

void MSELossFwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for mse_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  int64_t reduction = inputs[1].toInt();

  ns_MSELossKernel::Params param = synapse_mse_loss_params_builder(reduction);
  p_context_->params_.emplace<ns_MSELossKernel::Params>(param);
  p_context_->params_size_ = sizeof(param);

  Tensor output;
  if (reduction == at::Reduction::Reduction::None) {
    output = at::empty(self.sizes(), self.options());
  } else {
    output = at::empty({1}, self.options());
  }

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
  LOG_FUNC_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "mse_loss_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  MSELossFwdOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&self, &target};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(reduction)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  if (reduction != at::Reduction::Reduction::None) {
    // Note: pytorch expects 0d tensor (scalar)
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  LOG_FUNC_END;
  return out.at(0);
}

void MSELossBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for mse_loss operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  int64_t reduction = inputs[1].toInt();

  ns_MSELossKernel::Params param = synapse_mse_loss_params_builder(reduction);
  p_context_->params_.emplace<ns_MSELossKernel::Params>(param);
  p_context_->params_size_ = sizeof(param);

  auto output = at::empty(self.sizes(), self.options());

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
  LOG_FUNC_BEGIN;

  // Convert 0D tensor to 1D tensor before passing to Synapse
  if (grad_output.dim() == 0) {
    grad_output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "mse_loss_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  MSELossBwdOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&grad_output, &self, &target};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(reduction)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  LOG_FUNC_END;
  return out.at(0);
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) ->(Tensor output, Tensor total_weight)")
                .impl_unboxedOnlyKernel<
                    decltype(nll_loss_forward_hpu),
                    &nll_loss_forward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(nll_loss_backward_hpu),
                    &nll_loss_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(mse_loss_forward_hpu),
                    &mse_loss_forward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(mse_loss_backward_hpu),
                    &mse_loss_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));