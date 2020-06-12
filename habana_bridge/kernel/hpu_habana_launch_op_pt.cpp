/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <algorithm>
#include <sstream>
#include <unordered_map>

#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
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
#include "habana_bridge/kernel/hpu_habana_meta_op_list.h"
#include "synapse_helpers/graph_builder/graph_build_context.h"
#include "synapse_helpers/graph_builder/shape_adjust.h"
#include "habana_device/tensor_builder.h"
#include "habana_helpers/tensor_utils.h"

using namespace torch::jit;

std::ostream &operator<< (std::ostream &O, const RecipeArgumentSpec &v) {
  O << v.hash_code << '\n';
  return O;
}

std::ostream &operator<< (std::ostream &O, const RecipeValueSpec &v) {
  O << "recipe addr : " << v.recipe.get() << ' ';
  if (v.syn_tensor_names != nullptr && v.syn_tensor_buffers != nullptr) {
    O << "(";
    size_t j = 0;
    for (auto & i : *v.syn_tensor_names) { O << " " << i << ':' << v.syn_tensor_buffers->at(j++); }
    O << " )";
  }
  O << '\n';

  return O;
}

std::ostream &operator<< (std::ostream &O, const RecipeCacheSimple &v) {
  std::cout << "number of recipes : " << v.map_.size() <<'\n';
  for (auto & i: v.map_) {
    O << "-------------------" << '\n';
    O << "key :: " << *(i.first);
    O << "-------------------" << '\n';
    O << "val :: " << i.second;
    O << "-------------------" << '\n';
  }
  return O;
}

HabanaLaunchOpPT::HabanaLaunchOpPT(const torch::jit::Node* node, bool debug) {
  subgraph_ = node->g(attr::Subgraph);
  opname_ = node->kind().toQualString();
  debug_ = debug;
}

habana::LayoutFormat getPTTensorLayout() {
  return habana::LayoutFormat::NCHW;
}

habana::LayoutFormat HabanaLaunchOpPT::getTensorChannelOrder(
    torch::jit::Value* val) {
  // The value of the node keeps the tensor physical layout memorized
  // We can update this later if we see any changes to the way layouts are
  // handled
  TORCH_CHECK(value_to_tensor_layout.find(val) != std::end(value_to_tensor_layout), " HabanaFusion : Channel order not updated");
  return value_to_tensor_layout[val];
}

// See if we are in any leagally accepted channel orders
bool HabanaLaunchOpPT::isChannelOrderSupported(
    torch::jit::Value* val,
    const habana::LayoutFormat &supported_channel_order) {
  return (supported_channel_order == habana::LayoutFormat::ANY)
      || (supported_channel_order == getTensorChannelOrder(val));
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

void HabanaLaunchOpPT::GetSynapseInputs(
    const HabanaOperatorPtr &habana_op,
    torch::jit::Node* node) {
  auto node_ins = node->inputs();
  for (const auto value_in : node_ins) {
    if (value_to_ivalue[value_in] && value_to_ivalue[value_in]->isTensor()) {
      auto pt_tensor = value_to_ivalue[value_in]->toTensor();

      // Find if an input tensor is already mapped
      // NB: It seems Habana doesn't support shared input to
      // different nodes in graph
      auto is_already_mapped = pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
          std::end(pt_to_synapse_tensors);

      if (is_already_mapped) {
        auto syn_tensor_input = pt_to_synapse_tensors.find(value_to_ivalue[value_in]);
        auto &syn_tensor = habana_op->SetSynapseInput(std::move(syn_tensor_input->second));
        pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);
      } else {
        auto &syn_tensor = habana_op->AllocateSynapseInput(
            *syn_graph_ptr, &pt_tensor, true);

        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);
        input_names.push_back(syn_tensor.tensor_name_);
        input_buffers.push_back(pt_tensor.data_ptr());

      }
    }
  }
}

void HabanaLaunchOpPT::GetSynapseOutputs(
    const HabanaOperatorPtr &habana_op,
    torch::jit::Node* node) {
    auto output_tensors_pt = habana_op->GetOutputs();
    auto &output_tensors_syn = habana_op->GetSynOutputs();
    auto &excluded_out_indices = habana_op->GetSynOutputIndicesExcludedInNode();
    auto output_nodes = node->outputs();
    auto habana_kernel_meta_data = habana_op->GetKernelMetaData();
    habana::LayoutFormat out_layout;
    int output_nodes_idx = 0, output_tensor_idx = 0;
    TORCH_CHECK(output_nodes.size() == output_tensors_pt.size() - excluded_out_indices.size(),
                "HabanaFusionOp Lowering: Number of output nodes generated doesnt match the graph");
    for (synapse_helpers::tensor &out_tensor_syn : output_tensors_syn) {
      if (excluded_out_indices.find(output_tensor_idx) == excluded_out_indices.end()) {
        value_to_ivalue[output_nodes[output_nodes_idx]] = new IValue(output_tensors_pt[output_tensor_idx]);
        //Get the layout from the kernels, this has to be passed from kernel meta data which is WIP.
        try
        {
          out_layout = habana_kernel_meta_data.output_layout.at(output_tensor_idx);
        }
        catch (const std::out_of_range & ex)
        {
          out_layout = habana::LayoutFormat::ANY;
        }
        //TODO : we can check what format to fill in case of ANY. as it may be channel last
        value_to_tensor_layout[output_nodes[output_nodes_idx]]
          = out_layout == habana::LayoutFormat::ANY ? habana::LayoutFormat::NCHW : out_layout;
        pt_to_synapse_tensors.emplace(
                value_to_ivalue[output_nodes[output_nodes_idx]], out_tensor_syn);
        output_nodes_idx++;
      }
    output_names.push_back(out_tensor_syn.tensor_name_);
    output_buffers.push_back(output_tensors_pt[output_tensor_idx].data_ptr());
    output_tensor_idx++;
  }
}


at::IntArrayRef getDimsForLayout(habana::LayoutFormat channel_order) {
  at::IntArrayRef dims;
  if(channel_order == habana::LayoutFormat::NCHW) {
     dims = {0, 3, 1, 2};
  } else if(channel_order == habana::LayoutFormat::NHWC) {
     dims = {0, 2, 3, 1};
  } else if(channel_order == habana::LayoutFormat::HWCK) {
     dims = {2, 3, 1, 0};
  } else {
     TORCH_CHECK(" Habana Fusion op permute called for unsupported channel order");
  }
  return dims;
}
// For now, we permute tensors at graph leaves once
// THis function permutes a given tensor to desired layout and modifies
// input_tensor list to have the new tensor
at::Tensor HabanaLaunchOpPT::permuteTensor(
    torch::jit::Value* value_in,
    const at::Tensor &input,
    habana::LayoutFormat permute_order) {

  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  HabanaOperatorPtr permute_kernel = habana::CreateHabanaOperator(
      device_id, "aten::permute", input.scalar_type());
  TORCH_CHECK(
      permute_kernel != nullptr,
      " \n Permute kernel isnt supported in graph mode ");

  habana_kernels.push_back(permute_kernel);
  //set input synapse tensors
  auto is_already_mapped = pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
      std::end(pt_to_synapse_tensors);

  if (is_already_mapped) {
    auto syn_tensor_input = pt_to_synapse_tensors.find(value_to_ivalue[value_in]);
    auto &syn_tensor = permute_kernel->SetSynapseInput(std::move(syn_tensor_input->second));
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);
  } else {
    auto pt_tensor = value_to_ivalue[value_in]->toTensor();
    auto &syn_tensor = permute_kernel->AllocateSynapseInput(
        *syn_graph_ptr, &pt_tensor, true);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);
    input_names.push_back(syn_tensor.tensor_name_);
    input_buffers.push_back(pt_tensor.data_ptr());
  }

  auto dims = getDimsForLayout(permute_order);

  torch::jit::Stack input_stack = {IValue(input), IValue(dims)};
  // setup the config params for the kernels
  bool persistent = isInGraphOutputs(value_in);
  permute_kernel->AllocateAndAddSynapseNode(*syn_graph_ptr, input_stack, persistent);
  auto outputs_permute = permute_kernel->GetOutputs();

  //set output synapse tensor
  auto& output_tensors_syn = permute_kernel->GetSynOutputs();
  for (synapse_helpers::tensor &out_tensor_syn : output_tensors_syn) {
    // make the output of permute the input for next synapse kernel
    // permute has a single output
    value_to_ivalue[value_in] = new IValue(outputs_permute[0]);
    value_to_tensor_layout[value_in] = permute_order;
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(
            value_to_ivalue[value_in], out_tensor_syn);
    if(persistent) {
        output_names.push_back(out_tensor_syn.tensor_name_);
        output_buffers.push_back(outputs_permute[0].data_ptr());
    }
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
    torch::jit::Node* node,
    const HabanaOperatorPtr &habana_kernel) {
  //Get the metadata for all inputs, used for preprocessing inputs
  auto &habana_kernel_meta_data = habana_kernel->GetKernelMetaData();
  // Check if its ok to change teh input tensor in the graph attached to value
  auto node_ins = node->inputs();
  int tensor_idx = 0;
  habana::LayoutFormat in_layout;
  for (const auto value_in : node_ins) {
    // TODO:If a tensor is permuted in first iteration and thats same
    // everytime, like weight in conv We need a way to remember that and used
    // pre-permuted memory/tensor in next iterations
    if(value_to_ivalue[value_in])
    {
        try
        {
          in_layout = habana_kernel_meta_data.input_layout.at(tensor_idx);
        }
        catch (const std::out_of_range & ex)
        {
          in_layout = habana::LayoutFormat::ANY;
        }
        // TODO: For channel_last order, we need not do the permute but need to change
        // the input tensor size, stride as done in habana_helpers::change_tensors_to_memory_format.
        // Need to check the input tensor memory_format to drive this.
        if ((value_in->type()->kind() == c10::TypeKind::TensorType) &&
            !(isChannelOrderSupported(value_in, in_layout))) {
            //permute
            permuteTensor(
                value_in,
                value_to_ivalue[value_in]->toTensor(),
                in_layout);
            tensor_idx++;
        }
    }
    // TODO : add checks for doing flattening/slicing anything that is
    // required.
  }
}

void HabanaLaunchOpPT::postProcessOutputs() {
  // Do we need a optimization pass here? What should we look for?
  for (auto node : subgraph_->nodes()) {
    auto node_outs = node->outputs();
    for (const auto value_out : node_outs) {
      // TODO : If a tensor is permuted in first iteration and thats same
      // everytime, like weight in conv We need a way to remember that and
      // used pre-permuted memory/tensor in next iterations
      if (value_to_ivalue[value_out] && value_out->type()->kind() == c10::TypeKind::TensorType &&
          getTensorChannelOrder(value_out) != habana::LayoutFormat::NCHW &&
          isInGraphOutputs(value_out)) {
        permuteTensor(
            value_out,
            value_to_ivalue[value_out]->toTensor(),
            habana::LayoutFormat::NCHW);
      }
      // TODO : add checks for doing flattening/slicing anything that is
      // required.
    }
  }
}

void HabanaLaunchOpPT::handlePrimNodes(torch::jit::Node* node)
{
  TORCH_CHECK(node->kind() == torch::jit::prim::Constant,
              " Habana Fusion only supports constant type prim nodes");
  auto node_vals = node->outputs();
  for (const auto value : node_vals) {
    auto val = new IValue(toIValue(value).value());
    if (val->isNone()) {
      continue;
    }
    value_to_ivalue[value] = val;
  }
}

torch::jit::Stack HabanaLaunchOpPT::getStackForNode(torch::jit::Node* node) {
  torch::jit::Stack stack_in;
  auto node_inputs = node->inputs();
  for (auto input : node_inputs) {
    if(value_to_ivalue[input])
        stack_in.insert(stack_in.end(), *value_to_ivalue[input]);
    else
        stack_in.insert(stack_in.end(), IValue());

  }
  return stack_in;
}

c10::ScalarType HabanaLaunchOpPT::getNodeScalarType(torch::jit::Node* node) {
  //return the data type of first input tensor
  for (auto input : node->inputs())
    {
      if (value_to_ivalue[input] && value_to_ivalue[input]->isTensor()) {
        return value_to_ivalue[input]->toTensor().scalar_type();
      }
    }
  //Default return float for now if no tensor found
  return c10::ScalarType::Float;
}

void HabanaLaunchOpPT::handleMetaOps(torch::jit::Node* node) {
  //Call the meta op via CPU impl
  //Some ops dont support c10 op.callBoxed so we need to call via JIT
  torch::jit::Stack stack;
  void *in_data, *out_data;
  auto node_ins = node->inputs();
  for (const auto value_in : node_ins) {
    stack.insert(stack.end(), *value_to_ivalue[value_in]);
    if(value_to_ivalue[value_in]->isTensor())
    {
      auto tensor = value_to_ivalue[value_in]->toTensor();
      if(pt_to_synapse_tensors.find(value_to_ivalue[value_in]) ==
        std::end(pt_to_synapse_tensors))
        {
            in_data = tensor.data_ptr();
            auto dtype =  tensor.scalar_type();
            meta_syn_tensors.push_back(habana_helpers::create_tensor(tensor,
                                      syn_graph_ptr->get_graph_handle(),
                                      true, dtype));
            pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], meta_syn_tensors.back());
            input_names.push_back(meta_syn_tensors.back().tensor_name_);
            input_buffers.push_back(tensor.data_ptr());
        }

    }
  }
  torch::jit::Operator jit_op = node->getOperator();
  auto offset = jit_op.getOperation()(stack);

  auto syn_tensor_input = pt_to_synapse_tensors.find(value_to_ivalue[node_ins[0]]);
  TORCH_CHECK(offset == 0);
  auto node_outs = node->outputs();
  auto outputs = last(stack, node_outs.size());
  int i = 0;
  for(const auto val_out : node_outs) {
    IValue *ival = new IValue;
    *ival = outputs[i];
    value_to_ivalue[val_out] = ival;
    if(ival->isTensor())
    {
      value_to_tensor_layout[val_out] = getPTTensorLayout();

      auto tensor = ival->toTensor();
      auto dtype =  tensor.scalar_type();
      //create a tensor variant on the same memory section as the input
      auto variant =  synapse_helpers::tensor_builder(
                        tensor.sizes(),
                        habana_helpers::pytorch_to_synapse_type(dtype))
                        .mark_persistence(true)
                        .with_memory_section(syn_tensor_input->second.memorysection())
                        .build(
                        synapse_helpers::HPURegistrar::get_device(
                        tensor.device().index()),
                        syn_tensor_input->second.graph());

      meta_syn_tensors.push_back(absl::get<synapse_helpers::tensor>(std::move(variant)));
      auto &syn_tensor = meta_syn_tensors.back();

      pt_to_synapse_tensors.emplace(value_to_ivalue[val_out], syn_tensor);
      output_names.push_back(syn_tensor.tensor_name_);
      output_buffers.push_back(tensor.data_ptr());
      out_data = tensor.data_ptr();
    }
    i++;
  }
  TORCH_CHECK(in_data == out_data, "HabanaFusion : Data pointer changed in Meta op");
}

void HabanaLaunchOpPT::CompileAndExecuteHabanaFusedOpKernel() {
  LOG_FUNC_BEGIN;

  // figure out the right device id
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  std::ostringstream ostream;
  ostream << opname_ << '_' << graph_id;
  graph_id++;
  synapse_helpers::graph syn_graph = habana_helpers::create_graph(device_id, ostream.str().c_str());
  syn_graph_ptr = &syn_graph;

  // for each node in IR graph, at this point the graph is a list with nodes
  // topoloically sorted
  // TODO : check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = subgraph_->nodes();
  for (auto* node : graph_nodes) {

    //Prim nodes require special handling and are a special case
    if(node->kind().is_prim())
    {
      handlePrimNodes(node);
      continue;
    }

    //if its a meta op we need to call the CPU impl and capture changes
    //only valid for single tensor ops
    //Can we avoid the string match here?
    if(HabanaMetaOpList::isHabanaMetaOp(node->kind().toQualString()))
    {
      handleMetaOps(node);
      continue;
    }
    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::CreateHabanaOperator(
        device_id, node->kind().toQualString(), getNodeScalarType(node));

    TORCH_CHECK(
        HabanaKernel != nullptr, std::string(" \n  kernel ")
                                 + std::string(node->kind().toQualString())
                                 + std::string(" isnt supported in graph mode "));

    // See if we need to modify/permute tesnors
    processInputs(node, HabanaKernel);

    // Create/attach the synapse inputs from aten tensors
    GetSynapseInputs(HabanaKernel, node);

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

  postProcessOutputs();

  // set outputs to output structure
  std::shared_ptr<synapse_helpers::graph::recipe_handle> synh_recipe;
  if (CompileSynapseGraph(synh_recipe)) {
    RecipeValueSpec rv(synh_recipe);
    // reoroder the input_names and input_buffers
    input_names.clear();
    input_buffers.clear();

    for (size_t i = pt_stack->size()-num_inputs; i < pt_stack->size(); i++) {
      torch::jit::IValue *input_ptr = &(pt_stack->at(i));
      auto it = pt_to_synapse_tensors.find(input_ptr);
      if (it != std::end(pt_to_synapse_tensors)) {
        auto &syn_tensor = it->second;
        input_names.push_back(syn_tensor.tensor_name_);
        input_buffers.push_back(input_ptr->toTensor().data_ptr());
      }
      else {
        TORCH_CHECK(false && "synapse tensor not found");
      }
    }

    TORCH_CHECK(input_names.size() == input_buffers.size());
    TORCH_CHECK(output_names.size() == output_buffers.size());

    rv.syn_tensor_names = std::make_shared<std::vector<std::string>> (input_names);
    rv.syn_tensor_names->insert(rv.syn_tensor_names->end(), output_names.begin(), output_names.end());

    rv.syn_tensor_buffers = std::make_shared<std::vector<void *>> (input_buffers);
    rv.syn_tensor_buffers->insert(rv.syn_tensor_buffers->end(), output_buffers.begin(), output_buffers.end());

    rv.aten_outputs = std::make_shared<std::vector<torch::jit::IValue*>>(std::vector<torch::jit::IValue*>());
    for (auto output : subgraph_->outputs()) {
      rv.aten_outputs->push_back(value_to_ivalue[output]);
    }

    LaunchRecipe(rv);

    // Update the stack
    drop(*pt_stack, num_inputs);
    for (auto output : subgraph_->outputs()) {
      pt_stack->insert(pt_stack->end(), *value_to_ivalue[output]);
    }

    // Add the <key,value> pair to the map
    std::shared_ptr<RecipeArgumentSpec> ra_spec =
      std::make_shared<RecipeArgumentSpec>(false, input_refs, subgraph_);

    //recipe_cache.map_.emplace(ra_spec, rv);
    recipe_cache.add(ra_spec, rv);
  }
  else {
    TORCH_CHECK(false && "synapse graph compilation failed");
  }

  LOG_FUNC_END;
}

bool HabanaLaunchOpPT::CompileSynapseGraph(
    std::shared_ptr<synapse_helpers::graph::recipe_handle>& synh_recipe) {
  auto compile_result = syn_graph_ptr->compile();
  synh_recipe = get_value(std::move(compile_result));
  return (synh_recipe != nullptr);
}

void HabanaLaunchOpPT::LaunchRecipe(RecipeValueSpec &rv) {
  rv.SelfCheck();
  std::shared_ptr<synapse_helpers::graph::recipe_handle> last_recipe = rv.recipe;

  std::vector<synLaunchTensorInfo> syn_launch_info;
  syn_launch_info.reserve(rv.syn_tensor_names->size());

  for (size_t i = 0; i < rv.syn_tensor_names->size(); ++i)
    syn_launch_info.emplace_back(synLaunchTensorInfo{
        rv.syn_tensor_names->at(i).c_str(), reinterpret_cast<uint64_t>(rv.syn_tensor_buffers->at(i))});

  auto & device = synapse_helpers::HPURegistrar::get_device();

  synStreamHandle stream_handle = device.get_compute_stream();

  synapse_helpers::graph::launch_info ln_info(last_recipe->device_);
  synapse_helpers::graph::create_launch_info(ln_info, *last_recipe);
  synapse_helpers::graph::launch(ln_info, *last_recipe, syn_launch_info);

  TORCH_HABANA_CHECK(synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
}

void HabanaLaunchOpPT::clear() {
  pt_stack = nullptr;
  habana_kernels.clear();
  input_names.clear();
  output_names.clear();
  input_buffers.clear();
  output_buffers.clear();
  value_to_tensor_layout.clear();
  pt_to_synapse_tensors.clear();
}

bool HabanaLaunchOpPT::IsCached(std::shared_ptr<RecipeArgumentSpec> &spec) {
  bool is_found(false);
  std::cout << '\n';
  if (!recipe_cache.empty()) {
    if (recipe_cache.exists(spec)) {
      std::cout << "       -------------" << '\n';
      std::cout << "       | cache hit |" << '\n';
      std::cout << "       -------------" << '\n';
      is_found = true;
    }
    else {
      std::cout << "       --------------" << '\n';
      std::cout << "       | cache miss |" << '\n';
      std::cout << "       --------------" << '\n';
    }
  }
  else {
    std::cout << "       ------------------" << '\n';
    std::cout << "       | first iteration |" << '\n';
    std::cout << "       ------------------" << '\n';
  }
  std::cout << '\n';

  return is_found;
}

void HabanaLaunchOpPT::run(torch::jit::Stack& stack) {
  LOG_FUNC_BEGIN;
  num_inputs = subgraph_->inputs().size();
  auto subgraph_inputs = subgraph_->inputs();
  input_refs = last(stack, num_inputs);

  // Keep a handle to the stack for future use
  pt_stack = &stack;

  // Fusion pass should ensure all nodes are on Habana, if all nodes not on
  // habana device, we should assert
  bool is_all_hpu = true;
  for (auto &input : input_refs) {
    is_all_hpu = input.toTensor().device().type() != c10::DeviceType::HABANA
        ? false
        : is_all_hpu;
  }

  // We dont support running some ops on CPU while running fused op on Habana
  // All tensors should be alocated to habana before entering this phase
  TORCH_CHECK(is_all_hpu == true, " Habana Fusion needs all tensors to be in HPU ");

  // caching :: begin
  if (enable_caching) {
    std::shared_ptr<RecipeArgumentSpec> spec =
      std::make_shared<RecipeArgumentSpec>(false, input_refs, subgraph_);

    if (IsCached(spec)) {
      // This is cache hit. Run the cached recipe
      RecipeValueSpec rv = recipe_cache.get(spec);

      // Patch the input buffers
      std::shared_ptr<std::vector<void *>> buffers = rv.syn_tensor_buffers;
      size_t i = 0;
      for (auto const &input : input_refs) {
        buffers->at(i++) = input.toTensor().data_ptr();
      }

      LaunchRecipe(rv);

      // Update the stack from the recipe itself
      drop(*pt_stack, num_inputs);
      for (auto ival_ptr : *(rv.aten_outputs)) {
        pt_stack->insert(pt_stack->end(), *ival_ptr);
      }
      clear();
      return;
    }
    // caching :: end
  }

  {
    size_t i = 0;
    size_t j = stack.size()-num_inputs;
    for ( ; j < stack.size(); j++) {
      auto value_input = subgraph_inputs[i];
      value_to_ivalue[value_input] = &stack[j];
      value_to_tensor_layout[value_input] = getPTTensorLayout();
      i++;
    }
  }

  //<Decription> This is the main function that
  //  a. creates the HabanaLaunchOp
  //  b. compiles and executes the same
  CompileAndExecuteHabanaFusedOpKernel();

  // clear the context
  // TODO : See if we need to add a contect to this object pointer or clearing like this is good?
  clear();

  LOG_FUNC_END;
}
