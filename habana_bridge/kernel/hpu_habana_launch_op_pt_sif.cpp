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

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <ATen/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_device/HPUAllocator.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/tensor_builder.h"

#include "habana_bridge/kernel/ds_graph_recompile.h"
#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_bridge/kernel/refinement_engine.h"
#include "habana_bridge/passes/hpu_habana_persistence_marker_pass.h"

#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"

#include "habana_kernels/hccl_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/unary_kernels.h"

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"

#include "hpu_ops/hpu_op_helper.h"

#include "pytorch_helpers/util/jitgraph_utils.h"
#include "synapse_helpers/env_flags.h"

using namespace torch::jit;
using namespace jitgraph_utils;

namespace habana {

synapse_helpers::tensor& HabanaLaunchOpPT::allocate_synapse_tensor(
    at::Tensor& pt_tensor,
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph) {
  auto& syn_tensor =
      habana_op->AllocateSynapseInput(syn_graph, pt_tensor, true);
  return syn_tensor;
}

torch::jit::Stack HabanaLaunchOpPT::create_stack_for_node(
    const torch::jit::Node* node,
    bool& flag,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  torch::jit::Stack node_stack;
  for (auto ni_val : node->inputs()) {
    if (val_to_ival_map.count(ni_val) == 0) {
      flag = false;
      continue;
    }
    node_stack.push_back(val_to_ival_map[ni_val]);
  }
  return node_stack;
}

void HabanaLaunchOpPT::create_synapse_input(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  std::vector<at::Tensor> pt_tensor_list;
  const auto& ival = val_to_ival_map[value_in];
  if (ival.isTensor()) {
    PT_TEST_DEBUG(
        "For %",
        value_in->debugName(),
        " adding syn_tensor, tensor ",
        habana_helpers::DebugString(ival.toTensor()));
    pt_tensor_list.emplace_back(ival.toTensor());
  } else {
    PT_TEST_DEBUG("For %", value_in->debugName(), " adding following");
    const auto& ival_list = ival.toListRef();
    for (const auto& ival_elem : ival_list) {
      if (!ival_elem.isNone()) {
        PT_TEST_DEBUG(
            " syn_tenosr, tensor",
            habana_helpers::DebugString(ival_elem.toTensor()));
        pt_tensor_list.emplace_back(ival_elem.toTensor());
      }
    }
  }
  for (auto& pt_tensor : pt_tensor_list) {
    if (!pt_tensor.defined()) {
      continue;
    }
    auto& syn_tensor = allocate_synapse_tensor(pt_tensor, habana_op, syn_graph);
    PT_TEST_DEBUG(
        "Allocated synpase tensor for input tensor: ", syn_tensor.id());
  }
}

void HabanaLaunchOpPT::create_synapse_inputs(
    torch::jit::Node* node,
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  int input_idx = 0;
  for (const auto value_in : node->inputs()) {
    auto value_exists = val_to_ival_map.find(value_in);
    HABANA_ASSERT(value_exists != std::end(val_to_ival_map));
    auto ivalue = value_exists->second;
    if (ivalue.isTensor()) {
      PT_TEST_DEBUG("Input coming from %", value_in->debugName());
      create_synapse_input(value_in, habana_op, syn_graph, val_to_ival_map);
    } else if (value_in->node()->kind() == torch::jit::prim::ListConstruct) {
      PT_TEST_DEBUG(
          "Input coming from ListConstruct output %", value_in->debugName());
      HABANA_ASSERT(ivalue.isTensorList(), "TensorList expected");
      PT_TEST_DEBUG("Tensorlist found for input %", value_in->debugName());
      auto prev_node = value_in->node();
      for (auto& prev_value_in : prev_node->inputs()) {
        PT_TEST_DEBUG("Checking prev_value_in %", prev_value_in->debugName());
        if (val_to_ival_map.count(prev_value_in)) {
          HABANA_ASSERT(
              val_to_ival_map[prev_value_in].isTensor(),
              "Input to ListConstruct can not be TensorList");
          create_synapse_input(
              prev_value_in, habana_op, syn_graph, val_to_ival_map);
        }
      }
    } else {
      PT_TEST_DEBUG(
          "Currently unsupported ivalue for %", value_in->debugName());
    }
    input_idx += 1;
  }
}

int64_t HabanaLaunchOpPT::get_output_tensors_count(
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph) {
  std::deque<synapse_helpers::tensor_or_ref>& syn_outputs =
      habana_op->GetSynOutputs();
  std::deque<synapse_helpers::tensor_or_ref>& syn_inputs =
      habana_op->GetSynInputs();
  int64_t output_count = syn_outputs.size();
  int int_shape_tensor_count = 0;
  if (syn_graph.is_dynamic_graph()) {
    for (synapse_helpers::tensor& in_tensor_syn : syn_inputs) {
      if (in_tensor_syn.is_intermediate_shape_tensor()) {
        HABANA_ASSERT(in_tensor_syn.is_shape_tensor());
        int_shape_tensor_count++;
      }
    }
  }
  output_count += int_shape_tensor_count;

  return output_count;
}

OutputMetaDataVector HabanaLaunchOpPT::populate_node_output_metadata(
    torch::jit::Node* node) {
  OutputMetaDataVector output_metadata{};
  // tensorList and Unpack pair is supported
  if (node->output(0)->type() == torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    node = unpack_node;
  }

  auto node_outs = node->outputs();
  for (auto value_out : node_outs) {
    OutputMetaData md(*value_out);
    output_metadata.emplace_back(md);
  }
  return output_metadata;
}

void HabanaLaunchOpPT::process_outputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map,
    std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map) {
  auto output_nodes = node->outputs();

  if (node->output(0)->type() == torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node = GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    output_nodes = unpack_node->outputs();
  }

  size_t output_idx = 0;
  auto currentSifTensorIdx = habana::ShapeInference::GetSifTensorId();
  for (auto& out_tensor_pt : habana_op->GetOutputs()) {
    val_to_ival_map.emplace(
        output_nodes[output_idx], torch::jit::IValue(out_tensor_pt));
    tidx_to_tensor_map.insert({currentSifTensorIdx, out_tensor_pt});
    output_idx++;
    currentSifTensorIdx++;
  }
}

void HabanaLaunchOpPT::print_stack(torch::jit::Stack& st) {
  PT_TEST_DEBUG("stack.size=", st.size(), ", details::");
  for (size_t idx = 0; idx < st.size(); idx++) {
    PT_TEST_DEBUG(habana_helpers::DebugString(st.at(idx)));
  }
}

void HabanaLaunchOpPT::print_val_to_ival_map(
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  PT_TEST_DEBUG(
      "\nval_to_ival_map_begin",
      "\n size=",
      val_to_ival_map.size(),
      ",  details::");
  for (auto p : val_to_ival_map) {
    PT_TEST_DEBUG(
        "%",
        p.first->debugName(),
        " -> ",
        habana_helpers::DebugString(p.second));
  }
  PT_TEST_DEBUG("val_to_ival_map_end");
}

void HabanaLaunchOpPT::print_graph_outputs(
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  PT_TEST_DEBUG("\nGraph Outputs");
  for (auto graph_output : jit_ir_graph->outputs()) {
    HABANA_ASSERT(
        val_to_ival_map.count(graph_output),
        "Output for %",
        graph_output->debugName(),
        " is missing from val_to_ival_map");
    PT_TEST_DEBUG(
        "%",
        graph_output->debugName(),
        " -> ",
        habana_helpers::DebugString(val_to_ival_map[graph_output]));
  }
}

void HabanaLaunchOpPT::print_tidx_to_tensor_map(
    const std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map) {
  PT_TEST_DEBUG("\nAfter hybrid output sif pass tidx_to_tensor_map");
  std::vector<size_t> tidx_vec;
  for (auto const& p : tidx_to_tensor_map) {
    tidx_vec.emplace_back(p.first);
  }

  std::sort(tidx_vec.begin(), tidx_vec.end());
  for (auto const& idx : tidx_vec) {
    PT_TEST_DEBUG(
        "sif_tidx : ",
        idx,
        habana_helpers::DebugString(tidx_to_tensor_map.at(idx)));
  }
}

void HabanaLaunchOpPT::visit_prim_node(
    const torch::jit::Node* node,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  if (torch::jit::prim::Constant == node->kind()) {
    for (const auto value : node->outputs()) {
      HABANA_ASSERT(val_to_ival_map.count(value) == 0);
      val_to_ival_map[value] = IVal(toIValue(value).value());
    }
  } else if (torch::jit::prim::ListConstruct == node->kind()) {
    std::vector<at::Tensor> tensorList;
    for (const auto input : node->inputs()) {
      HABANA_ASSERT(val_to_ival_map.count(input));
      auto input_ival = val_to_ival_map[input];
      HABANA_ASSERT(input_ival.isTensor());
      tensorList.emplace_back(input_ival.toTensor());
    }
    auto node_outputs = node->outputs();
    HABANA_ASSERT(node_outputs.size() == 1);
    val_to_ival_map[node_outputs[0]] = IVal(tensorList);
  }
}

void HabanaLaunchOpPT::RunHybridSif(
    std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map) {
  PT_BRIDGE_BEGIN;

  PT_TEST_DEBUG(
      "\nRunning hybrid shape inference on graph: ", GetSynapseGraphName());
  habana::PrintStack(*pt_stack);
  PT_TEST_DEBUG(
      "JIT_IR_Graph_BEGIN\n", jit_ir_graph->toString(), "JIT_IR_Graph_END\n");

  std::unordered_map<CValPtr, torch::jit::IValue> val_to_ival_map;
  auto graph_inputs = jit_ir_graph->inputs();
  TORCH_CHECK(input_refs.size() == graph_inputs.size(), "Input size mismatch");
  for (size_t i = 0; i < graph_inputs.size(); i++) {
    auto input = graph_inputs[i];
    val_to_ival_map[input] = input_refs[i];
  }

  // Figure out the right device id
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();

  auto syn_graph =
      habana_helpers::create_graph(device.id(), GetSynapseGraphName(), true);
  syn_graph.set_dynamic_graph(true);

  std::vector<at::Tensor> input_shape_tensors_vec;
  for (auto* node : jit_ir_graph->nodes()) {
    // print_val_to_ival_map(val_to_ival_map);
    std::string op_name(node->kind().toQualString());

    PT_TEST_DEBUG("\nVisiting:", op_name);

    // There should not be any meta ops
    HABANA_ASSERT(
        HabanaMetaOpList::isHabanaMetaOp(node->kind().toQualString()) == false,
        "Can not process meta op");

    // Prim nodes require special handling and are a special case
    if (node->kind().is_prim()) {
      PT_TEST_DEBUG("Constant found");
      visit_prim_node(node, val_to_ival_map);
      continue;
    }

    // TODO: visit restride nodes
    if ((strcmp(op_name.c_str(), "hpu::restride_cl") == 0) ||
        (strcmp(op_name.c_str(), "hpu::restride") == 0)) {
      PT_TEST_DEBUG("Restride found, skipping ...");
      continue;
    }

    // Get kernel context
    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr habana_op =
        KernelRegistry().get(device_id, op, getNodeScalarType(node));

    TORCH_CHECK(habana_op, op, " isn't registered in KernelRegistry!");

    bool is_mapped_flag{true};
    auto op_input_stack =
        create_stack_for_node(node, is_mapped_flag, val_to_ival_map);
    HABANA_ASSERT(
        is_mapped_flag, "Cannot proceed with unmapped input for ", op_name);

    // print_stack(op_input_stack);

    // Collect input shape tensors, To add them at last after graph inputs
    for (auto const& input : op_input_stack) {
      if (input.isTensor()) {
        auto tensor = input.toTensor();
        auto impl = habana_lazy::GetHbInternalTensorImpl(tensor);
        if (impl && impl->isShapeTensor()) {
          input_shape_tensors_vec.emplace_back(tensor);
        }
      }
    }

    auto propagate_shape{[&]() -> void {
      // Non ComputeOutputShape based path, adjust SifTensrorId
      PT_TEST_DEBUG(
          "Using non ComputeOutputShape based flow. Going to add tpc kernel ",
          habana_op->GetGuid(),
          " for ",
          op_name);

      // Create the synapse inputs from aten tensors
      create_synapse_inputs(node, habana_op, syn_graph, val_to_ival_map);

      // Setup the config params for the kernels
      auto outputs_metadata = populate_node_output_metadata(node);

      habana_op->AllocateAndAddSynapseNode(
          syn_graph, op_input_stack, outputs_metadata);

      process_outputs(habana_op, node, val_to_ival_map, tidx_to_tensor_map);

      auto output_count = get_output_tensors_count(habana_op, syn_graph);
      habana::ShapeInference::IncrementSifTensorId(output_count);
      PT_TEST_DEBUG_TH(
          "After increment: sif tensor id = ",
          habana::ShapeInference::GetSifTensorId());
    }};

    auto output_shape_info = habana_op->ComputeOutputShape(op_input_stack);
    if (output_shape_info.empty()) {
      PT_TEST_DEBUG_TH("ComputeOutputShape is not supported for ", op_name);
      propagate_shape();
    } else {
      // Output shape info based flow
      PT_TEST_DEBUG_TH(
          "Using ComputeOutputShape shape info based flow for ",
          habana_op->GetGuid(),
          ", ",
          op_name);
      auto output_tensors = output_shape_info.GetOutputTensor();

      // Collect all output tensors
      for (auto& t : output_tensors) {
        tidx_to_tensor_map.insert({std::get<0>(t), std::get<1>(t)});
      }

      // Recursivly collect all shape tensors
      std::vector<IdxTensorTup> intermediate_shape_tensor_cs;
      ProcessShapeTensorsCS(output_shape_info, intermediate_shape_tensor_cs);

      // Get all values of shape tensor
      for (auto& t : intermediate_shape_tensor_cs) {
        tidx_to_tensor_map.insert({std::get<0>(t), std::get<1>(t)});
      }

      HABANA_ASSERT(node->outputs().size() == output_tensors.size());
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        auto output = node->outputs().at(i);
        HABANA_ASSERT(val_to_ival_map.count(output) == 0);
        val_to_ival_map[output] = IVal(std::get<1>(output_tensors[i]));
      }
    }
  }

  // For all Graph inputs create a sif mapping
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    HABANA_ASSERT(input_refs[i].isTensor());
    auto inp_sif_tid = habana::ShapeInference::ReadAndIncrementSifTensorId();
    tidx_to_tensor_map.insert({inp_sif_tid, input_refs[i].toTensor()});
    PT_TEST_DEBUG_TH(
        "For input tensors, adding to tidx_to_tensor_map: ",
        inp_sif_tid,
        " -> ",
        habana_helpers::DebugString(input_refs[i].toTensor()));
  }

  // For all input shape tensors create a sif mapping
  for (auto const& input_tensor : input_shape_tensors_vec) {
    auto inp_sif_tid = habana::ShapeInference::ReadAndIncrementSifTensorId();
    tidx_to_tensor_map.insert({inp_sif_tid, input_tensor});
    PT_TEST_DEBUG_TH(
        "For input shape tensors, adding to tidx_to_tensor_map: ",
        inp_sif_tid,
        " -> ",
        habana_helpers::DebugString(input_tensor));
  }

  // For debugging
  // print_val_to_ival_map(val_to_ival_map);
  // print_graph_outputs(val_to_ival_map);
  // print_tidx_to_tensor_map(tidx_to_tensor_map);

  PT_BRIDGE_END;
}

} // namespace habana
