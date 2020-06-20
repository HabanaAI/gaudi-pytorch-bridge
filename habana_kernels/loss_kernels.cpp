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
  PT_KERNEL_BEGIN;

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

  PT_KERNEL_END;
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
  PT_KERNEL_BEGIN;
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

  PT_KERNEL_END;
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
  PT_KERNEL_BEGIN;

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

  PT_KERNEL_END;
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
  PT_KERNEL_BEGIN;

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

  PT_KERNEL_END;
  return out.at(0);
}

// Input tensors
// 1	Gradient             FP32/FP16/BF16	2D
// 2	Weights              FP32	2D
// 3	Moments              FP32	2D
// 4	Indices              I32	1D
// 5	Learning rate	       FP32	1D
// 6	Valid count	         I32	1D
//
// Output tensors
// 1	Weights              FP32/FP16/BF16	2D
// 2	Moments              FP32	2D
#if 0 // TODO: TPC kernel seems to give wrong results.
std::tuple<torch::Tensor, torch::Tensor>
optimizer_sparse_sgd_with_valid_count_hpu(
    torch::Tensor gradients,
    torch::Tensor weights_in,
    torch::Tensor moments_in,
    torch::Tensor indices,
    torch::Tensor learning_rate,
    int64_t valid_count,
    float mom,
    bool nesterov) {
  LOG_FUNC_BEGIN;
  std::cout
      << "Inside New Op :: optimizer_sparse_sgd_with_valid_count_hpu valid_count = "
      << valid_count << std::endl;
  ns_OptimizerSparseSGD::Params params;
  params.mom = mom;
  params.nesterov = nesterov;
  auto weights_out = at::empty(weights_in.sizes(), weights_in.options());
  auto moments_out = at::empty(moments_in.sizes(), moments_in.options());
  weights_out.copy_(weights_in, false);
  moments_out.copy_(moments_in, false);
  auto cast_indices = habana_helpers::cast_tensor_to_integer(indices);
  auto long_tensor = at::empty({1}, indices.options());
  auto lcpu = long_tensor.to("cpu");
  int64_t* lptr = static_cast<int64_t*>(lcpu.data_ptr());
  *lptr = valid_count;
  auto valid_count_tensor =
      lcpu.to(c10::ScalarType::Int).to(long_tensor.device());
  std::vector<const Tensor*> pt_inputs;
  pt_inputs.push_back(&gradients);
  pt_inputs.push_back(&weights_in);
  pt_inputs.push_back(&moments_in);
  pt_inputs.push_back(&cast_indices);
  pt_inputs.push_back(&learning_rate);
  pt_inputs.push_back(&valid_count_tensor);
  std::vector<const Tensor*> pt_outputs{&weights_out, &moments_out};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "optimizer_sparse_sgd_with_valid_count_2d_",
      &params,
      sizeof(params),
      SynapsePassType::NO_PASS_WITH_TYPE);
  LOG_FUNC_END;
  return std::tie(weights_out, moments_out);
}
#else
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
optimizer_sparse_sgd_with_valid_count_cpu(
    torch::Tensor gradients,
    torch::Tensor weights_in,
    torch::Tensor moments_in,
    torch::Tensor indices,
    torch::Tensor learning_rate,
    int64_t valid_count,
    float mom,
    bool nesterov) {
  /*
  moments_out[sparse_indices] = momentum_in[sparse_indices] * state.mom +
                                gradients[sparse_indices];
  gradients_out[sparse_indices] = momentum_out[sparse_indices];
  weights_out[sparse_indices] =
    weights_in[sparse_indices] - state.lr * gradients_out[sparse_indices];
  */
  auto sizes = weights_in.sizes().vec();
  float* gp = static_cast<float*>(gradients.data_ptr());
  float* winp = static_cast<float*>(weights_in.data_ptr());
  float* minp = static_cast<float*>(moments_in.data_ptr());
  Tensor weights_out = at::empty(weights_in.sizes(), weights_in.options());
  Tensor moments_out = at::empty(moments_in.sizes(), moments_in.options());
  Tensor grad_output = at::empty(gradients.sizes(), gradients.options());
  weights_out.copy_(weights_in, false);
  moments_out.copy_(moments_in, false);
  grad_output.copy_(gradients, false);
  float* woutp = static_cast<float*>(weights_out.data_ptr());
  float* moutp = static_cast<float*>(moments_out.data_ptr());
  float* goutp = static_cast<float*>(grad_output.data_ptr());
  int* inp = static_cast<int*>(indices.data_ptr());
  float* lrp = static_cast<float*>(learning_rate.data_ptr());
  float gtemp;
  unsigned vec_len = sizes[1];
  for (unsigned i = 0; i < valid_count; i++) {
    for (unsigned k = 0; k < vec_len; k++) {
      // momentum update
      moutp[inp[i] * vec_len + k] =
          minp[inp[i] * vec_len + k] * mom + gp[inp[i] * vec_len + k];
      gtemp = moutp[inp[i] * vec_len + k];
      // grad update
      if (nesterov) {
        goutp[inp[i] * vec_len + k] = gp[inp[i] * vec_len + k] + mom * gtemp;
      } else {
        goutp[inp[i] * vec_len + k] = gtemp;
      }
      // weight update
      woutp[inp[i] * vec_len + k] = winp[inp[i] * vec_len + k] - *lrp * gtemp;
    }
  }
  return std::make_tuple(weights_out, moments_out, grad_output);
}

std::tuple<torch::Tensor, torch::Tensor>
optimizer_sparse_sgd_with_valid_count_hpu(
    torch::Tensor gradients,
    torch::Tensor weights_in,
    torch::Tensor moments_in,
    torch::Tensor indices,
    torch::Tensor learning_rate,
    int64_t valid_count,
    float mom,
    bool nesterov) {
  PT_KERNEL_BEGIN;
  auto sizes = weights_in.sizes().vec();
  for (unsigned int i = 0; i < weights_in.dim(); i++)
    PT_KERNEL_DEBUG("sizes = ", sizes[i]);
  auto cast_indices = habana_helpers::cast_tensor_to_integer(indices);
  auto hpu = indices.device();
  auto result = optimizer_sparse_sgd_with_valid_count_cpu(
      gradients.to("cpu"),
      weights_in.to("cpu"),
      moments_in.to("cpu"),
      cast_indices.to("cpu"),
      learning_rate.to("cpu"),
      valid_count,
      mom,
      nesterov);
  auto ret1 = std::get<0>(result);
  auto ret2 = std::get<1>(result);
  PT_KERNEL_END;
  return std::make_tuple(ret1.to(hpu), ret2.to(hpu));
}

#endif
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
