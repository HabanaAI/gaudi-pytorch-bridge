/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <algorithm>
#include "habana_device/HPUCheck.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include <unordered_map>
#include "habana_kernels/kernel_utils.h"

using namespace torch::jit;

HabanaLaunchOpPT::HabanaLaunchOpPT(const torch::jit::Node* node, bool debug) {
  subgraph_ = node->g(attr::Subgraph);
  debug_ = debug;
}

habana::LayoutFormat HabanaLaunchOpPT::getTensorChannelOrder(
    torch::jit::Value* val) {
  // The value of the node keeps the tensor physical layout memorized
  // We can update this later if we see any changes to the way layouts are
  // handled
  TORCH_CHECK(value_to_tensor_layout.find(val) != value_to_tensor_layout.end(), "Value pointer doesnt have a valid tensor Layout");
  return value_to_tensor_layout[val];
}

// See if we are in any leagally accepted channel orders
bool HabanaLaunchOpPT::isChannelOrderSupported(
    torch::jit::Value* val,
    const habana::LayoutFormat &supported_channel_order) {
  return (supported_channel_order == habana::LayoutFormat::ANY)
      || (supported_channel_order == getTensorChannelOrder(val));
}

void HabanaLaunchOpPT::GetSynapseInputs(
    const HabanaOperatorPtr &habana_op,
    synapse_helpers::graph& graph,
    torch::jit::Node* node) {
  auto node_ins = node->inputs();
  for (const auto value_in : node_ins) {
    if (value_to_ivalue[value_in].isTensor()) {
      auto pt_tensor = value_to_ivalue[value_in].toTensor();

      // Find if an input tensor is already mapped
      // NB: It seems Habana doesn't support shared input to
      // different nodes in graph
      auto is_already_mapped = pt_to_synapse_tensors.find(&value_to_ivalue[value_in]) !=
          std::end(pt_to_synapse_tensors);

      if (is_already_mapped) {
        auto syn_tensor_input = pt_to_synapse_tensors.find(&value_to_ivalue[value_in]);
        auto &syn_tensor = habana_op->SetSynapseInput(std::move(syn_tensor_input->second));
        pt_to_synapse_tensors.erase(&value_to_ivalue[value_in]);
        pt_to_synapse_tensors.emplace(&value_to_ivalue[value_in], syn_tensor);
      } else {
        auto &syn_tensor = habana_op->AllocateSynapseInput(
            graph, &pt_tensor, true);

        pt_to_synapse_tensors.emplace(&value_to_ivalue[value_in], syn_tensor);
        input_names.push_back(syn_tensor.tensor_name_);
        input_buffers.push_back(pt_tensor.data_ptr());

      }
    }
  }
}

void HabanaLaunchOpPT::GetSynapseOutputs(
    const HabanaOperatorPtr &habana_op,
    torch::jit::Node* node) {

    auto &habana_kernel_meta_data = habana_op->GetKernelMetaData();
    auto output_tensors_pt = habana_op->GetOutputs();
    auto &output_tensors_syn = habana_op->GetSynOutputs();
    auto output_nodes = node->outputs();
    int i = 0;
    TORCH_CHECK(output_nodes.size() == output_tensors_pt.size(), "HabanaFusionOp Lowering: Number of output nodes generated doesnt match the graph");
    for (auto &out_tensor_syn : output_tensors_syn) {
      value_to_ivalue[output_nodes[i]] = IValue(output_tensors_pt[i]);
      //Get the layout from the kernels, this has to be passed from kernel meta data which is WIP.
      value_to_tensor_layout[output_nodes[i]] = habana::LayoutFormat::NCHW /*habana_kernel_meta_data.output_layout[i]*/;
      pt_to_synapse_tensors.emplace(
              &value_to_ivalue[output_nodes[i]], out_tensor_syn);
      output_names.push_back(out_tensor_syn.tensor_name_);
      output_buffers.push_back(output_tensors_pt[i].data_ptr());
    i++;
  }
}


at::IntArrayRef getDimsForLayout(habana::LayoutFormat channel_order)
{
  at::IntArrayRef dims;
  if(channel_order == habana::LayoutFormat::NCHW)
     dims = {0, 3, 1, 2};
  else if(channel_order == habana::LayoutFormat::NHWC)
     dims = {0, 2, 3, 1};
  else
      TORCH_CHECK(" Habana Fusion op permute called for unsupported channel order");
  return dims;
}
// For now, we permute tensors at graph leaves once
// THis function permutes a given tensor to desired layout and modifies
// input_tensor list to have the new tensor
at::Tensor HabanaLaunchOpPT::permuteTensor(
    synapse_helpers::graph& syn_graph,
    torch::jit::Value* value_in,
    const at::Tensor &input,
    habana::LayoutFormat permute_oder) {

  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  HabanaOperatorPtr permute_kernel = habana::CreateHabanaOperator(
      device_id, "aten::permute", input.scalar_type());
  TORCH_CHECK(
      permute_kernel != nullptr,
      " \n Permute kernel isnt supported in graph mode ");

  habana_kernels.push_back(permute_kernel);
  //set input synapse tensors
  auto is_already_mapped = pt_to_synapse_tensors.find(&value_to_ivalue[value_in]) !=
      std::end(pt_to_synapse_tensors);

  if (is_already_mapped) {
    auto syn_tensor_input = pt_to_synapse_tensors.find(&value_to_ivalue[value_in]);
    auto &syn_tensor = permute_kernel->SetSynapseInput(std::move(syn_tensor_input->second));
    pt_to_synapse_tensors.erase(&value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(&value_to_ivalue[value_in], syn_tensor);
  } else {
    auto pt_tensor = value_to_ivalue[value_in].toTensor();
    auto &syn_tensor = permute_kernel->AllocateSynapseInput(
        syn_graph, &pt_tensor, true);
    pt_to_synapse_tensors.emplace(&value_to_ivalue[value_in], syn_tensor);
    input_names.push_back(syn_tensor.tensor_name_);
    input_buffers.push_back(pt_tensor.data_ptr());
  }

  auto dims = getDimsForLayout(permute_oder);

  torch::jit::Stack input_stack = {IValue(input), IValue(dims)};
  // setup the config params for the kernels
  permute_kernel->AllocateAndAddSynapseNode(syn_graph, input_stack, false);
  auto outputs_permute = permute_kernel->GetOutputs();

  //set output synapse tensor
  auto &output_tensors_syn = permute_kernel->GetSynOutputs();
  for (auto &out_tensor_syn : output_tensors_syn) {
    // make the output of permute the input for next synapse kernel
    // permute has a single output
    value_to_ivalue[value_in] = IValue(outputs_permute[0]);
    value_to_tensor_layout[value_in] = permute_oder;
    pt_to_synapse_tensors.erase(&value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(
            &value_to_ivalue[value_in], out_tensor_syn);
  }
  return outputs_permute[0];
}

bool HabanaLaunchOpPT::isInGraphInputs(torch::jit::Value* value) {
  auto graph_ins = subgraph_->inputs();
  for (auto value_in : graph_ins) {
    if (value->unique() == value_in->unique()) {
      return true;
    }
  }
  return false;
}

void HabanaLaunchOpPT::processInputs(
    synapse_helpers::graph& syn_graph,
    torch::jit::Node* node,
    const HabanaOperatorPtr &habana_kernel) {

  //Get the metadata for all inputs, used for preprocessing inputs
  auto &habana_kernel_meta_data = habana_kernel->GetKernelMetaData();
  // Check if its ok to change teh input tensor in the graph attached to value
  auto node_ins = node->inputs();
  TORCH_CHECK(node_ins.size() == habana_kernel_meta_data.input_layout.size(), "HabanaFusionOp : Error in Habana Kernel Meta data");
  int tensor_idx = 0;
  for (const auto value_in : node_ins) {
    // TODO:If a tensor is permuted in first iteration and thats same
    // everytime, like weight in conv We need a way to remember that and used
    // pre-permuted memory/tensor in next iterations
    if ((value_in->type()->kind() == c10::TypeKind::TensorType) &&
        !(isChannelOrderSupported(value_in, habana_kernel_meta_data.input_layout[tensor_idx]))) {
        //permute
        permuteTensor(
            syn_graph,
            value_in,
            value_to_ivalue[value_in].toTensor(),
            habana_kernel_meta_data.input_layout[tensor_idx]);
        tensor_idx++;
    }
    // TODO : add checks for doing flattening/slicing anything that is
    // required.
  }
}

bool HabanaLaunchOpPT::isInGraphOutputs(torch::jit::Value* value) {
  auto graph_outs = subgraph_->outputs();
  for (auto value_out : graph_outs) {
    if (value->unique() == value_out->unique()) {
      return true;
    }
  }
  return false;
}

void HabanaLaunchOpPT::postProcessOutputs(synapse_helpers::graph& syn_graph) {
  // Do we need a optimization pass here? What should we look for?
  for (auto node : subgraph_->nodes()) {
    auto node_outs = node->outputs();
    for (const auto value_out : node_outs) {
      // TODO:If a tensor is permuted in first iteration and thats same
      // everytime, like weight in conv We need a way to remember that and
      // used pre-permuted memory/tensor in next iterations
      if (value_out->type()->kind() == c10::TypeKind::TensorType &&
          getTensorChannelOrder(value_out) != habana::LayoutFormat::NCHW &&
          isInGraphOutputs(value_out)) {
        permuteTensor(
            syn_graph,
            value_out,
            value_to_ivalue[value_out].toTensor(),
            habana::LayoutFormat::NCHW);
      }
      // TODO : add checks for doing flattening/slicing anything that is
      // required.
    }
  }
}

torch::jit::Stack HabanaLaunchOpPT::getStackForNode(torch::jit::Node* node) {
  torch::jit::Stack stack_in;
  auto inputs = node->inputs();
  for (auto input : inputs) {
    stack_in.insert(stack_in.end(), value_to_ivalue[input]);
  }
  return stack_in;
}

c10::ScalarType HabanaLaunchOpPT::getNodeScalarType(torch::jit::Node* node)
{
  //return the data type of first input tensor
  for (auto input : node->inputs())
    {
      if(value_to_ivalue[input].isTensor())
        return value_to_ivalue[input].toTensor().scalar_type();
    }
  //Default return float for now if no tensor found
  return c10::ScalarType::Float;
}
void HabanaLaunchOpPT::compile() {

  // figure out the right device id
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  //TODO : get the correct name from the graph
  synapse_helpers::graph syn_graph =
      habana_helpers::create_graph(device_id, "habana_op"/*subgraph_->block()->owningNode()->kind().toQualString()*/);

  // for each node in IR graph, at this point the graph is a list with nodes
  // topoloically sorted
  // TODO: check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = subgraph_->nodes();
  for (auto* node : graph_nodes) {
    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::CreateHabanaOperator(
        device_id, node->kind().toQualString(), getNodeScalarType(node));

    TORCH_CHECK(
        HabanaKernel != nullptr, std::string(" \n  kernel ")
                                 + std::string(node->kind().toQualString())
                                 + std::string(" isnt supported in graph mode "));

    // See if we need to modify/permute tesnors
    processInputs(syn_graph, node, HabanaKernel);

    // Create/attach the synapse inputs from aten tensors
    GetSynapseInputs(HabanaKernel, syn_graph, node);

    torch::jit::Stack input_stack = getStackForNode(node);

    // setup the config params for the kernels
    HabanaKernel->AllocateAndAddSynapseNode(syn_graph, input_stack, true);

    // Get the output tensors created back from the kernel
    // We set type so that the created tensor is propagated throughout graph
    GetSynapseOutputs(HabanaKernel, node);

    //Adding to a vector as we share context through shared pointers and we dont want to
    //call delete untill we are done with whole graph
    habana_kernels.push_back(HabanaKernel);
  }

  postProcessOutputs(syn_graph);


  // set outputs to output structure
  // API call : compile and execute synapse graph
  // TODO: Pass to RunCached function
  std::shared_ptr<synapse_helpers::graph::recipe_handle> pRecipe;
  if (CompileSynapseGraph(syn_graph, pRecipe)) {
    auto syn_launch_info = habana_helpers::generate_syn_launch_tensor_info(
        input_names, input_buffers, output_names, output_buffers);

    auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

    synStreamHandle stream_handle = device.get_compute_stream();

    synapse_helpers::graph::launch_info launchInfo(pRecipe->device_);
    syn_graph.create_launch_info(launchInfo, *pRecipe);
    syn_graph.launch(launchInfo, *pRecipe, syn_launch_info);
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

bool HabanaLaunchOpPT::CompileSynapseGraph(
    synapse_helpers::graph& synGraph,
    std::shared_ptr<synapse_helpers::graph::recipe_handle>& recipeId) {
  auto compile_result = synGraph.compile();
  recipeId = get_value(std::move(compile_result));
  if (recipeId == nullptr) {
    return false;
  }
  return true;
}

habana::LayoutFormat getPTTensorLayout() {
  return habana::LayoutFormat::NCHW;
}

void HabanaLaunchOpPT::CreateHabanaFusedOpKernel() {
  LOG_FUNC_BEGIN;

  // Invocation compile and execute
  // Add cache later
  compile();

  //clear the context, see if we need to add a contect to this object pointer
  // or clearing like this is good?
  clear();
  LOG_FUNC_END;
}

void HabanaLaunchOpPT::clear()
{
  habana_kernels.clear();
  input_names.clear();
  output_names.clear();
  input_buffers.clear();
  output_buffers.clear();
  value_to_tensor_layout.clear();
  pt_to_synapse_tensors.clear();
}

void HabanaLaunchOpPT::run(torch::jit::Stack& stack) {
  LOG_FUNC_BEGIN;
  //LF;
  int num_inputs = subgraph_->inputs().size();
  auto subgraph_inputs = subgraph_->inputs();
  inputs = last(stack, num_inputs);

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto value_input = subgraph_inputs[i];
    value_to_ivalue[value_input] = inputs[i];
    value_to_tensor_layout[value_input] = getPTTensorLayout();
  }

  // Fusion pass should ensure all nodes are on Habana, if all nodes not on
  // habana device, we should assert
  bool is_all_hpu = true;
  for (auto &input : inputs) {
    is_all_hpu = input.toTensor().device().type() != c10::DeviceType::HABANA
        ? false
        : is_all_hpu;
  }

  // We dont support running some ops on CPU while running fused op on Habana
  // All tensors should be alocated to habana before entering this phase
  TORCH_CHECK(is_all_hpu == true, " Habana Fusion needs all tensors to be in HPU ");

  //<Decription> This is the main function that creates the HabanaLaunchOp and
  // calls it
  // All code flow is encapsulated in it
  CreateHabanaFusedOpKernel();

  // Update the stack
  drop(stack, num_inputs);
  auto outputs = subgraph_->outputs();
  for (auto output : outputs) {
    stack.insert(stack.end(), value_to_ivalue[output]);
  }
  LOG_FUNC_END;
}
