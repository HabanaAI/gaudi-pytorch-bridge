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
#include <ATen/InferSize.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

void CompareOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for aten::gt Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for compare op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg 2 for compare op needs to be of tensor type");
  Tensor self = inputs[0].toTensor();
  Tensor other = inputs[1].toTensor();

  if (self.ndimension() != other.ndimension()) {
    //
    // we need to reshape tensor which has lesser dimesnions, reshape_tensor_idx
    // points to the tensor for which we need to reshape & input_tensor_idx
    // points to tensor which goes directly to compare kernel without reshape
    int32_t reshape_tensor_idx =
        (self.ndimension() > other.ndimension()) ? 1 : 0;
    int32_t input_tensor_idx = (self.ndimension() > other.ndimension()) ? 0 : 1;
    Tensor& reshape_tensor =
        (self.ndimension() > other.ndimension()) ? other : self;
    Tensor& input_tensor =
        (self.ndimension() > other.ndimension()) ? self : other;
    std::vector<int64_t> reshaped_sizes = std::vector<int64_t>(
        input_tensor.ndimension() - reshape_tensor.ndimension(), 1);
    auto reshape_tensor_sizes = reshape_tensor.sizes().vec();

    reshaped_sizes.insert(
        reshaped_sizes.end(),
        reshape_tensor_sizes.begin(),
        reshape_tensor_sizes.end());
    auto output = at::empty(
        reshaped_sizes,
        self.options().dtype(c10::ScalarType::Bool),
        self.suggest_memory_format());

    ReshapeOperator reshape(this->p_context_->device_id_, this->scalarType_);
    auto& reshape_in_syn_tensor = reshape.SetSynapseInput(
        std::move(p_context_->syn_inputs_[reshape_tensor_idx]));
    torch::jit::Stack stack = {IValue(reshape_tensor), IValue(reshaped_sizes)};
    reshape.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[reshape_tensor_idx] =
        std::move(reshape_in_syn_tensor);

    AllocateSynapseOutput(graph, output, is_output_persistent);
    synapse_helpers::tensor& reshape_out_syn_tensor =
        reshape.GetSynOutputs()[0];
    std::vector<synTensor> syn_inputs(2, nullptr);
    synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{output_syn_tensor.get()};
    synapse_helpers::tensor& input_syn_tensor =
        p_context_->syn_inputs_[input_tensor_idx];
    syn_inputs[input_tensor_idx] = input_syn_tensor.get();
    syn_inputs[reshape_tensor_idx] = reshape_out_syn_tensor.get();
    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        std::move(guid_));
  } else {
    auto output = at::empty(
        self.sizes(),
        self.options().dtype(c10::ScalarType::Bool),
        self.suggest_memory_format());
    AllocateSynapseOutput(graph, output, is_output_persistent);
    AddNodeToSynapseGraph(graph, nullptr, 0);
  }
}

/*************************************************************************
 * @brief Kernel implementation for aten.gt(self, other)
 * @param self - tensor_0
 * @param other - tensor_1
 ************************************************************************/
Tensor gt_hpu(Tensor& self, Tensor& other) {
  PT_KERNEL_BEGIN;
  size_t device_id = self.device().index();
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gt_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  //
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Create operator
  GtOperator Op(device_id, scalar_type);

  // Assign Inputs to the Operator
  std::vector<const at::Tensor*> pt_inputs{&self, &other};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(other)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;
  return out.at(0);
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema("aten::gt.Tensor(Tensor self, Tensor other) -> Tensor")
        .impl_unboxedOnlyKernel<decltype(gt_hpu), &gt_hpu>(
            DispatchKey::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));