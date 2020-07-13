/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/softmax_kernels.h"
#include "synapse_helpers/recipe.h"

#include <algorithm>

using namespace torch;

namespace habana {

void LogSoftmaxOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  bool half_to_float;
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for softmax operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input type expected to be int");
  if (!inputs[2].isNone())
    TORCH_CHECK(inputs[2].isBool(), "Input type expected to be Bool");

  at::Tensor self = inputs[0].toTensor();
  int dim = inputs[1].toInt();

  // FIXME Need to fix it for GraphMode SW-13887

  if (!inputs[2].isNone()) {
    half_to_float = inputs[2].toBool();
    TORCH_CHECK(
        !half_to_float,
        "softmax with half to float conversion is not supported on HPU");
  }

  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(self.ndimension() - 1 - dim)};

  p_context_->params_.emplace<ns_Softmax::Params>(params);
  p_context_->params_size_ = sizeof(params);

  auto output = at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void LogSoftmaxBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of input expected for softmax operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isInt(), "Input type expected to be int");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");

  at::Tensor grad = inputs[0].toTensor();
  at::Tensor output = inputs[1].toTensor();
  int dim = inputs[2].toInt();
  at::Tensor input = inputs[3].toTensor();

  dim = at::maybe_wrap_dim(dim, input.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(input.ndimension() - 1 - dim)};

  p_context_->params_.emplace<ns_Softmax::Params>(params);
  p_context_->params_size_ = sizeof(params);

  // For logsoftmax_bwd_ kernel, the node inputs are in order {grad, output,
  // input} The synapse graph needs only the grad and output, and i the order
  // {output, grad} The p_context_->pt_inputs_ abd p_context_->syn_inputs_ are
  // modified here to ensure this.

  TORCH_CHECK(
      p_context_->pt_inputs_.size() == 3,
      "logsoftmax_bwd node should have 3 input pytorch tensors");
  TORCH_CHECK(
      p_context_->syn_inputs_.size() == 3,
      "logsoftmax_bwd node should have 3 input synapse tensors");
  // Remove the "input" tensor at the end
  p_context_->pt_inputs_.pop_back();
  p_context_->syn_inputs_.pop_back();

  // Reorder the grad and output
  std::swap(p_context_->pt_inputs_[0], p_context_->pt_inputs_[1]);
  std::swap(p_context_->syn_inputs_[0], p_context_->syn_inputs_[1]);

  auto grad_output = at::empty(input.sizes(), input.options(), input.suggest_memory_format());

  AllocateSynapseOutput(graph, grad_output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/** log_softmax (forward pass) implementation for Habana device
 * @params [In] self: Input tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] half_to_float:
 */
Tensor log_softmax_hpu(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "logsoftmax_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // create the operator
  LogSoftmaxOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(half_to_float)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    std::vector<const at::Tensor*> inputs{&self};
    auto output = at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    std::vector<const at::Tensor*> pt_inputs{&self};
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

/** log_softmax (backward pass) implementation for Habana device
 * @params [In] grad: Backward pass Input tensor. 2-4D. bf16, fp32
 * @params [In] output: Forward pass Output tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] input: Forward pass Input tensor. 2-4D. bf16, fp32
 */
Tensor log_softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  PT_KERNEL_BEGIN;

  size_t device_id = grad.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = grad.scalar_type();
  std::string node_type =
      "logsoftmax_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // create the operator
  LogSoftmaxBackwardOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(grad), IValue(output), IValue(dim), IValue(input)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    std::vector<const at::Tensor*> pt_inputs{&output, &grad};
    auto output = at::empty(input.sizes(), input.options(), input.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    std::vector<const at::Tensor*> pt_inputs{&grad, &output, &input};
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}
} // end namespace habana

namespace habana {
SoftmaxOperator::SoftmaxOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator(
          "softmax_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
  this->CreateSynContext(device_id);
  kernel_meta_data_.input_layout.assign(
      {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
  kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
}

void SoftmaxOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for softmax operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input type expected to be int");
  TORCH_CHECK(inputs[2].isBool(), "Input type expected to be Bool");

  at::Tensor self = inputs[0].toTensor();
  int dim = inputs[1].toInt();
  bool half_to_float = inputs[2].toBool();

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(self.ndimension() - 1 - dim)};

  p_context_->params_.emplace<ns_Softmax::Params>(params);
  p_context_->params_size_ = sizeof(params);

  auto output = at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/** softmax (forward pass) implementation for Habana device
 * @params [In] self: Input tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] half_to_float:
 */
Tensor softmax_hpu(const Tensor& self, int64_t dim, const bool half_to_float) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");

  // FIXME need to add as a node to the graph to work for GraphMode
  // FIXME SW-13887
  // This part implements softmax.int
  Tensor self_casted;
  bool is_casted = false;
  if (self.scalar_type() == c10::ScalarType::Int ||
      self.scalar_type() == c10::ScalarType::Bool) {
    self_casted = habana_helpers::hpu_cast_tensor(
        self, at::scalarTypeToTypeMeta(c10::ScalarType::Float));
    is_casted = true;
  }

  size_t device_id = self.device().index();
  at::ScalarType scalar_type =
      is_casted ? self_casted.scalar_type() : self.scalar_type();
  std::string node_type =
      "softmax_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // create the operator
  SoftmaxOperator Op(device_id, scalar_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{is_casted ? &self_casted : &self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  std::vector<c10::IValue> stack = {
      is_casted ? IValue(self_casted) : IValue(self),
      IValue(dim),
      IValue(half_to_float)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}
} // end namespace habana

/** softmax (backward pass) implementation for Habana device
 * @params [In] grad: Backward pass Input tensor. 2-4D. bf16, fp32
 * @params [In] output: Forward pass Output tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] input: Forward pass Input tensor. 2-4D. bf16, fp32
 */
Tensor softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  PT_KERNEL_BEGIN;

  int64_t dim_ = at::maybe_wrap_dim(dim, input.dim(), /*wrap_scalar=*/true);

  auto input_grad = at::empty(input.sizes(), input.options(), input.suggest_memory_format());
  ns_Softmax::Params params{static_cast<int>(input.ndimension() - 1 - dim_)};

  std::vector<const at::Tensor*> pt_inputs{&output, &grad};
  std::vector<const at::Tensor*> pt_outputs{&input_grad};
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "softmax",
      &params,
      sizeof(params),
      SynapsePassType::BACKWARD_PASS);

  PT_KERNEL_END;
  return input_grad;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(habana::log_softmax_hpu),
                    &habana::log_softmax_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(habana::log_softmax_backward_hpu),
                    &habana::log_softmax_backward_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(habana::softmax_hpu),
                    &habana::softmax_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(softmax_backward_hpu),
                    &softmax_backward_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
