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
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/aten_hpu_type_default.h"
#include "habana_kernels/bitwise_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/simple_generic_kernel.h"

void NonZeroOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for NonZero operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");

  auto self = inputs[0].toTensor();
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  auto output_shape = DimVector{elements, dimensions};
  auto shape_tensor_shape = DimVector{5};
  // Create PT output stage 2
  auto cordinates_of_true = habana_helpers::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);
  auto shape_tensor = habana_helpers::createPTTensor(
      self,
      shape_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      true);

  std::vector<at::Tensor> outputs{cordinates_of_true, shape_tensor};
  HabanaOperator::SetPTOutputs(outputs);
}

void NonZeroOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for NonZero operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be tensor for NonZero operator");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "is_output_persistent expected to be vector of size 2");

  auto self = inputs[0].toTensor();
  size_t device_id = self.device().index();
  Tensor input_bool;
  BitwiseOrOutOperator Op(device_id, ScalarType::Bool);

  // tf_where_stage1 TPC requires bool input for now
  // support for Int, float, bf16 input without using gt, lt,
  // or operators dependent on JIRA
  // https://jira.habana-labs.com/browse/SW-36577
  if (self.scalar_type() != ScalarType::Bool) {
    at::ScalarType scalar_type = self.scalar_type();
    auto other = static_cast<ScalarType>(0);
    torch::jit::Stack stack{IValue(self), IValue(other)};

    // Check for values greater than 0
    GtOperator op_gt(device_id, scalar_type);
    auto& syn_arg1 =
        op_gt.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    op_gt.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[0] = std::move(syn_arg1);
    stack.clear();

    // Check for values less than 0
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(other));
    LtOperator op_lt(device_id, scalar_type);
    auto& syn_arg2 =
        op_lt.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
    op_lt.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[0] = std::move(syn_arg2);
    stack.clear();

    // create bool input tensor
    input_bool = habana_helpers::createPTTensor(
        self,
        self.sizes(),
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Bool,
        false);

    stack.emplace_back(IValue(input_bool));
    stack.emplace_back(IValue(op_gt.GetOutputs()[0]));
    stack.emplace_back(IValue(op_lt.GetOutputs()[0]));
    // Assign Inputs to the Operator
    Op.AllocateSynapseInput(graph, input_bool, false);
    Op.SetSynapseInput(std::move(op_gt.GetSynOutputs()[0]));
    Op.SetSynapseInput(std::move(op_lt.GetSynOutputs()[0]));
    Op.AllocateAndAddSynapseNode(graph, stack, false);
  }

  // Settings copied from tensorFlow file
  // tensorflow-training/habana_device/kernels/hpu_habana_where_op.h
  const uint tpc_count = 8;
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  auto intermediate_shape = DimVector{dimensions, elements};
  auto valid_shape = DimVector{2, tpc_count};
  auto output_shape = DimVector{elements, dimensions};
  auto shape_tensor_shape = DimVector{5};

  // create cordinates_unsqueezed and cordinates_valid Stage 1
  auto cordinates_unsqueezed = habana_helpers::createPTTensor(
      self,
      intermediate_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      false);
  auto cordinates_valid = habana_helpers::createPTTensor(
      self,
      valid_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      false);
  std::vector<synapse_helpers::tensor_or_ref> intermediateOutputs;
  intermediateOutputs.emplace_back(habana_helpers::create_tensor(
      cordinates_unsqueezed, graph.get_graph_handle(), false, c10::nullopt));
  intermediateOutputs.emplace_back(habana_helpers::create_tensor(
      cordinates_valid, graph.get_graph_handle(), false, c10::nullopt));

  ns_TfWhere::Params param;
  param.tpcCount = tpc_count;
  SetGuid("tf_where_stage1_fwd_i8");

  synapse_helpers::tensor& synStage1Output1 = intermediateOutputs[0];
  synapse_helpers::tensor& synStage1Output2 = intermediateOutputs[1];
  std::vector<synTensor> syn_in;
  std::vector<synTensor> syn_out{
      synStage1Output1.get(), synStage1Output2.get()};
  if (self.scalar_type() != ScalarType::Bool) {
    synapse_helpers::tensor& synInput1 = std::move(Op.GetSynOutputs()[0]);
    syn_in.push_back(synInput1.get());
  } else {
    synapse_helpers::tensor& synInput1 = p_context_->syn_inputs_[0];
    syn_in.push_back(synInput1.get());
  }
  // Where stage 1 node
  graph.add_node(
      std::move(syn_in),
      std::move(syn_out),
      &param,
      sizeof(param),
      std::move(guid_));

  // Create PT output stage 2
  auto cordinates_of_true = habana_helpers::createPTTensor(
      self,
      output_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent[0]);
  auto shape_tensor = habana_helpers::createPTTensor(
      self,
      shape_tensor_shape,
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      is_output_persistent[1]);
  // shape_tensor is of type UINT32 not supported by ScalarType, use synDataType
  synDataType synType = syn_type_uint32;
  AllocateSynapseOutput(graph, cordinates_of_true, is_output_persistent[0]);
  AllocateSynapseOutput(graph, shape_tensor, synType, is_output_persistent[1]);
  SetGuid("tf_where_stage2_fwd_i32");
  synapse_helpers::tensor& synStage2Output1 = p_context_->syn_outputs_[0];
  synapse_helpers::tensor& synStage2Output2 = p_context_->syn_outputs_[1];

  std::vector<synTensor> syn_in_stage2{
      synStage1Output1.get(), synStage1Output2.get()};
  std::vector<synTensor> syn_out_stage2{
      synStage2Output1.get(), synStage2Output2.get()};

  // Where stage 2 node
  graph.add_node(
      std::move(syn_in_stage2),
      std::move(syn_out_stage2),
      &param,
      sizeof(param),
      std::move(guid_));
}

/*************************************************************************
 * @brief Kernel implementation for torch.nonzero operator
 * @param self - Input tensor
 ************************************************************************/
Tensor nonzero_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();

  // Output required of type Int64
  if (self.numel() == 0) {
    auto shape = DimVector{0, dimensions};
    auto output = habana_helpers::createPTTensor(
        self,
        shape,
        self.options(),
        self.suggest_memory_format(),
        c10::ScalarType::Long,
        true);
    PT_KERNEL_END;
    return output;
  }
  Tensor self_in = self;
  // Long not supported for lt and gt operator
  // casting to int
  if (scalar_type == ScalarType::Long) {
    self_in = habana_helpers::cast_tensor_to_integer(self);
  }
  std::string node_type =
      "nonzero_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // Create the operator
  NonZeroOperator Op(device_id, self_in.scalar_type());
  std::vector<at::Tensor> pt_inputs{self_in};
  std::vector<c10::IValue> stack = {IValue(self_in)};

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");
  // Slice the output to select only relevant data
  // The shape tensor = [Dimension_detail, num_relevantElement, N/U, N/U, N/U]
  // Since output tensor[elements,Dimension] already has correct dimension
  // we need to slice along row to get relevant elements to outputs
  // i.e slice the output from where_stage_2 of shape (elementsxDimensions)
  // to get correct output (relevantElementsxDimensions)
  auto end = out.at(1)[1].item<int64_t>();
  if (end == 0) {
    out.at(0).unsafeGetTensorImpl()->set_sizes_and_strides(
        {0, dimensions}, {1, 1});
    PT_KERNEL_END;
    return out.at(0);
  }
  auto result = out.at(0).slice(0, 0, end, 1);
  // Remove this cast once index_put_ implementation using scatter_nd is
  // available
  auto output = habana_helpers::cast_tensor_to_long(result);
  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry = ::habana::KernelRegistry().add(
    "aten::nonzero",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<NonZeroOperator>(device_id, node_type);
    });