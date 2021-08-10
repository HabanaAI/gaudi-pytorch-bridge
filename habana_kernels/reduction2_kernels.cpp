/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
// #include <ATen/native/TensorIterator.h> // TODO: fix this include
#include <bitset>

#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/reduction2_kernels.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;
using namespace habana;

void Reduce2Operator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  Tensor self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  UNUSED bool keepdim = inputs[2].toBool();

  // wrap dim to positive value
  auto dim = c10::maybe_wrap_dim(dim_, self.dim(), true);
  auto out_shape = self.sizes().vec();
  out_shape[dim] = 1;
  ns_Reduction::Params params{};
  params.reductionDimension = self.dim() - dim - 1;
  auto output = habana_helpers::createPTTensor(
      self, out_shape, self.options(), is_output_persistent[0]);
  auto index = habana_helpers::createPTTensor(
      self,
      out_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent[1]);
  AllocateSynapseOutputs(graph, {output, index}, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void MaxDimOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for aten::max.dim operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for aten::max.dim operator");
  TORCH_CHECK(
      inputs[1].isInt(),
      "Input arg2 expected to be Int for aten::max.dim operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input arg3 expected to be Bool for aten::max.dim operator");

  Tensor self = inputs[0].toTensor();
  auto dim_ = inputs[1].toInt();
  UNUSED bool keepdim = inputs[2].toBool();
  auto dim = c10::maybe_wrap_dim(dim_, self.dim(), true);

  auto reduce_op =
      make_operator<Reduce2Operator>(self.device().index(), this->guid_);
  auto& syn_self = reduce_op->SetSynapseInput((p_context_->syn_inputs_[0]));
  std::vector<bool> reshapeadd{false, false};
  reduce_op->AllocateAndAddSynapseNode(
      graph, inputs, keepdim ? is_output_persistent : reshapeadd);
  p_context_->syn_inputs_[0] = std::move(syn_self);

  auto reshape_op =
      make_operator<ReshapeOperator>(self.device().index(), self.scalar_type());
  auto reshape_index =
      make_operator<ReshapeOperator>(self.device().index(), self.scalar_type());
  if (!keepdim) {
    auto out_shape = reduce_op->GetOutputs()[0].sizes().vec();
    out_shape.erase(out_shape.cbegin() + dim);
    reshape_op->SetSynapseInput((reduce_op->GetSynOutputs()[0]));
    torch::jit::Stack stack = {
        IValue(reduce_op->GetOutputs()[0]), IValue(out_shape)};
    reshape_op->AllocateAndAddSynapseNode(
        graph, stack, is_output_persistent[0]);
    stack.clear();

    reshape_index->SetSynapseInput((reduce_op->GetSynOutputs()[1]));
    stack = {IValue(reduce_op->GetOutputs()[1]), IValue(out_shape)};
    reshape_index->AllocateAndAddSynapseNode(
        graph, stack, is_output_persistent[1]);
    stack.clear();
  }

  synapse_helpers::tensor& reduce_op_syn_t =
      keepdim ? reduce_op->GetSynOutputs()[0] : reshape_op->GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(reduce_op_syn_t);

  auto reduce_op_pt_t =
      keepdim ? reduce_op->GetOutputs()[0] : reshape_op->GetOutputs()[0];
  p_context_->pt_outputs_.emplace_back(reduce_op_pt_t);

  synapse_helpers::tensor& out_syn_t = keepdim
      ? reduce_op->GetSynOutputs()[1]
      : reshape_index->GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(out_syn_t);

  auto out_pt_t =
      keepdim ? reduce_op->GetOutputs()[1] : reshape_index->GetOutputs()[0];
  p_context_->pt_outputs_.emplace_back(out_pt_t);
}

std::tuple<at::Tensor, at::Tensor> max_dim_hpu(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_max_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self), IValue(dim), IValue(keepdim)};
  // Create the operator
  MaxDimOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto out_shape = MaxDimOperator::compute_output_shape(
        self, c10::maybe_wrap_dim(dim, self.dim()), keepdim);
    Tensor output1, output2;
    if (out_shape.size() < 4) {
      output1 =
          at::empty(out_shape, self.options(), at::MemoryFormat::Contiguous);
      output2 = at::empty(
          out_shape,
          self.options().dtype(c10::ScalarType::Int),
          at::MemoryFormat::Contiguous);
    } else {
      output1 =
          at::empty(out_shape, self.options(), self.suggest_memory_format());
      output2 = at::empty(
          out_shape,
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format());
    }
    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{output1, output2};
    Op.SetPTOutputs(v);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});

    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::make_tuple(out.at(0), out.at(1));
}

synapse_helpers::tensor_or_ref MaxOperator::ReduceSingle(
    synapse_helpers::graph& graph,
    Tensor& input,
    int64_t i,
    synapse_helpers::tensor_or_ref syn_input) {
  Reduce2Operator reduce(input.device().index(), this->guid_);
  auto& reduce_syn_input = reduce.SetSynapseInput(std::move(syn_input));
  torch::jit::Stack stack = {IValue(input), IValue(i), IValue(true)};
  reduce.AllocateAndAddSynapseNode(graph, stack, {false, false});
  ReduceOpList.push_back(reduce);
  syn_input = std::move(reduce_syn_input);
  return syn_input;
}

void MaxOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for aten::max operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for aten::max operator");

  Tensor self = inputs[0].toTensor();
  p_context_->syn_inputs_[0] =
      ReduceSingle(graph, self, 0, std::move(p_context_->syn_inputs_[0]));
  for (auto i = 1; i < self.dim(); i++) {
    ReduceSingle(
        graph,
        ReduceOpList[i - 1].GetOutputs()[0],
        i,
        std::move(ReduceOpList[i - 1].GetSynOutputs()[0]));
  }

  auto reshape_op =
      make_operator<ReshapeOperator>(self.device().index(), self.scalar_type());
  std::vector<int64_t> out_shape{1};
  reshape_op->SetSynapseInput(
      std::move(ReduceOpList[self.dim() - 1].GetSynOutputs()[0]));
  torch::jit::Stack stack = {
      IValue(ReduceOpList[self.dim() - 1].GetOutputs()[0]), IValue(out_shape)};
  reshape_op->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(
      std::move(reshape_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(reshape_op->GetOutputs()[0]));
}

Tensor max_hpu(const at::Tensor& self) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reduce_max_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self)};
  // Create the operator
  MaxOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto out_shape = MaxOperator::compute_output_shape();
    auto output =
        at::empty(out_shape, self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{output};
    Op.SetPTOutputs(v);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
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

synapse_helpers::tensor_or_ref MinOperator::ReduceSingle(
    synapse_helpers::graph& graph,
    Tensor& input,
    int64_t i,
    synapse_helpers::tensor_or_ref syn_input) {
  Reduce2Operator reduce(input.device().index(), this->guid_);
  auto& reduce_syn_input = reduce.SetSynapseInput(std::move(syn_input));
  torch::jit::Stack stack = {IValue(input), IValue(i), IValue(true)};
  reduce.AllocateAndAddSynapseNode(graph, stack, {false, false});
  ReduceOpList.push_back(reduce);
  syn_input = std::move(reduce_syn_input);
  return syn_input;
}

void MinOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for aten::min operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for aten::min operator");

  Tensor self = inputs[0].toTensor();
  p_context_->syn_inputs_[0] =
      ReduceSingle(graph, self, 0, std::move(p_context_->syn_inputs_[0]));
  for (auto i = 1; i < self.dim(); i++) {
    ReduceSingle(
        graph,
        ReduceOpList[i - 1].GetOutputs()[0],
        i,
        std::move(ReduceOpList[i - 1].GetSynOutputs()[0]));
  }

  // Convert to 1D tensor for output
  auto reshape_op =
      make_operator<ReshapeOperator>(self.device().index(), self.scalar_type());
  std::vector<int64_t> out_shape{1};
  reshape_op->SetSynapseInput(
      std::move(ReduceOpList[self.dim() - 1].GetSynOutputs()[0]));
  torch::jit::Stack stack = {
      IValue(ReduceOpList[self.dim() - 1].GetOutputs()[0]), IValue(out_shape)};
  reshape_op->AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(
      std::move(reshape_op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(reshape_op->GetOutputs()[0]));
}

Tensor min_hpu(const at::Tensor& self) {
  PT_KERNEL_BEGIN;
  CONVERT_0D_TO_1D(self)
  at::ScalarType scalar_type = self.scalar_type();

  std::string node_type =
      "reduce_min_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {IValue(self)};
  // Create the operator
  MinOperator Op(device_id, scalar_type);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto out_shape = MinOperator::compute_output_shape();
    auto output =
        at::empty(out_shape, self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    std::vector<at::Tensor> v{output};
    Op.SetPTOutputs(v);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  CONVERT_1D_TO_0D(self, out.at(0))
  PT_KERNEL_END;
  return out.at(0);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::max_dim",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MaxDimOperator>(device_id, node_type);
            })
        .add(
            "aten::max",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MaxOperator>(device_id, node_type);
            })
        .add("aten::min", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<MinOperator>(device_id, node_type);
        });
