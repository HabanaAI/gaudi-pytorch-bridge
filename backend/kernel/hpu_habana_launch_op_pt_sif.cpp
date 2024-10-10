/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include <torch/csrc/jit/ir/constants.h>
#include <unordered_map>
#include "backend/helpers/graph.h"
#include "backend/jit_graph_cache.h"
#include "backend/jitgraph_utils.h"
#include "backend/kernel/ds_graph_recompile.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "backend/kernel/hpu_habana_launch_op_pt_sif_utils.h"
#include "backend/kernel/hpu_habana_meta_op_list.h"
#include "backend/kernel/hpu_shape_inference.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "hpu_ops/op_backend.h"

using namespace torch::jit;

namespace habana {

namespace {
synapse_helpers::tensor& allocate_synapse_tensor(
    at::Tensor& pt_tensor,
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph) {
  auto tmeta{get_tensor_extra_meta(pt_tensor, true)};
  if (tmeta && tmeta->is_shape_tensor()) {
    void* host_ptr = tmeta->get_compile_host_ptr();
    auto& syn_tensor = habana_op->AllocateSynapseInput(
        syn_graph, pt_tensor, true, tmeta->get_tensor_type(), host_ptr);
    return syn_tensor;
  } else {
    auto& syn_tensor =
        habana_op->AllocateSynapseInput(syn_graph, pt_tensor, true);
    return syn_tensor;
  }
}
} // namespace

torch::jit::Stack HabanaLaunchOpPT::create_stack_for_node(
    const torch::jit::Node* node,
    bool& flag,
    CValPtrtoIValueMap& val_to_ival_map) {
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

namespace {
void create_synapse_input(
    CValPtr value_in,
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph,
    CValPtrtoIValueMap& val_to_ival_map) {
  std::vector<at::Tensor> pt_tensor_list;
  const auto& ival = val_to_ival_map[value_in];
  if (ival.isTensor()) {
    PT_DYNAMIC_SHAPE_DEBUG(
        "For %",
        value_in->debugName(),
        " adding syn_tensor, tensor ",
        habana_helpers::DebugString(ival.toTensor()));
    pt_tensor_list.emplace_back(ival.toTensor());
  } else {
    PT_DYNAMIC_SHAPE_DEBUG("For %", value_in->debugName(), " adding following");
    const auto& ival_list = ival.toListRef();
    for (const auto& ival_elem : ival_list) {
      if (!ival_elem.isNone()) {
        PT_DYNAMIC_SHAPE_DEBUG(
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
    PT_DYNAMIC_SHAPE_DEBUG(
        "Allocated synapse tensor for input tensor: ", syn_tensor.id());
  }
}

void create_synapse_inputs(
    torch::jit::Node* node,
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph,
    CValPtrtoIValueMap& val_to_ival_map) {
  for (const auto value_in : node->inputs()) {
    auto value_exists = val_to_ival_map.find(value_in);
    HABANA_ASSERT(value_exists != std::end(val_to_ival_map));
    auto ivalue = value_exists->second;
    if (ivalue.isTensor()) {
      PT_DYNAMIC_SHAPE_DEBUG("Input coming from %", value_in->debugName());
      create_synapse_input(value_in, habana_op, syn_graph, val_to_ival_map);
    } else if (
        (value_in->node()->kind() == torch::jit::prim::ListConstruct) &&
        (ivalue.isTensorList())) {
      PT_DYNAMIC_SHAPE_DEBUG(
          "Input coming from ListConstruct output %", value_in->debugName());
      HABANA_ASSERT(ivalue.isTensorList(), "TensorList expected");
      PT_DYNAMIC_SHAPE_DEBUG(
          "Tensorlist found for input %", value_in->debugName());
      auto prev_node = value_in->node();
      for (auto& prev_value_in : prev_node->inputs()) {
        PT_DYNAMIC_SHAPE_DEBUG(
            "Checking prev_value_in %", prev_value_in->debugName());
        if (val_to_ival_map.count(prev_value_in)) {
          HABANA_ASSERT(
              val_to_ival_map[prev_value_in].isTensor(),
              "Input to ListConstruct can not be TensorList");
          create_synapse_input(
              prev_value_in, habana_op, syn_graph, val_to_ival_map);
        }
      }
    } else {
      PT_DYNAMIC_SHAPE_DEBUG(
          "Not creating synapse tensor for the ivalue for %",
          value_in->debugName());
    }
  }
}
} // namespace

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
    if (auto op = std::dynamic_pointer_cast<OpBackend>(habana_op)) {
      for (const auto& st : op->GetShapeTensors()) {
        if (st.is_intermediate_shape_tensor()) {
          HABANA_ASSERT(st.is_shape_tensor());
          int_shape_tensor_count++;
        }
      }
    }

    for (synapse_helpers::tensor& in_tensor_syn : syn_inputs) {
      if (in_tensor_syn.is_intermediate_shape_tensor()) {
        HABANA_ASSERT(in_tensor_syn.is_shape_tensor());
        int_shape_tensor_count++;
      }
    }
  }

  output_count +=
      int_shape_tensor_count + habana_op->GetSynImplicitOutputs().size();

  return output_count;
}

namespace {
OutputMetaDataVector populate_node_output_metadata(
    const torch::jit::Node* node) {
  OutputMetaDataVector output_metadata{};
  // If node output is tensor list
  // tensorList and Unpack pair is supported
  if (*node->output(0)->type() == *torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node =
        jitgraph_utils::GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    node = unpack_node;
  }

  auto node_outs = node->outputs();
  for (auto value_out : node_outs) {
    OutputMetaData md(*value_out);
    auto out_ptr = value_out->type()->cast<c10::TensorType>();
    if (out_ptr->scalarType().has_value()) {
      md.dtype = *out_ptr->scalarType();
    }
    output_metadata.emplace_back(md);
  }
  return output_metadata;
}
} // namespace

static at::Tensor GetDummyTensor(
    at::IntArrayRef sizes,
    at::ScalarType dtype = c10::ScalarType::Undefined) {
  const auto& t = at::detail::make_tensor<c10::TensorImpl>(
      c10::DispatchKeySet{at::DispatchKey::HPU, at::DispatchKey::AutogradHPU},
      c10::scalarTypeToTypeMeta(dtype),
      c10::Device(c10::kHPU, 0));
  t.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  return t;
}

namespace {
void process_shape_tensors(
    const HabanaOperatorPtr& habana_op,
    std::vector<at::Tensor>& intermediate_shape_tensors_vec) {
  // Auto gen op shape tensors
  if (auto op = std::dynamic_pointer_cast<OpBackend>(habana_op)) {
    for (const auto& st : op->GetShapeTensors()) {
      if (st.is_intermediate_shape_tensor()) {
        intermediate_shape_tensors_vec.emplace_back(
            GetDummyTensor(st.pt_shape()));
      }
    }
  }
  // Manual op shape tensors
  for (synapse_helpers::tensor& maybe_syn_shape_tensor :
       habana_op->GetSynInputs()) {
    if (maybe_syn_shape_tensor.is_intermediate_shape_tensor()) {
      intermediate_shape_tensors_vec.emplace_back(
          GetDummyTensor(maybe_syn_shape_tensor.pt_shape()));
    }
  }
  // Add shape tensors for all Operator created inside habanaOp
  std::vector<HabanaOperatorPtr> habana_kernels = habana_op->GetKernels();
  for (auto& habana_op : habana_kernels) {
    process_shape_tensors(habana_op, intermediate_shape_tensors_vec);
  }
}
} // namespace

void HabanaLaunchOpPT::process_outputs(
    const HabanaOperatorPtr& habana_op,
    torch::jit::Node* node,
    CValPtrtoIValueMap& val_to_ival_map,
    std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map) {
  auto output_nodes = node->outputs();

  if (node->output(0)->type() == torch::ListType::ofTensors() &&
      node->outputs().size() == 1) {
    auto unpack_node =
        jitgraph_utils::GetUnpackNodeFromTensorList(node->output(0));
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
    PT_DYNAMIC_SHAPE_DEBUG(
        "For node output, adding to tidx_to_tensor_map: ",
        currentSifTensorIdx,
        " -> ",
        habana_helpers::DebugString(out_tensor_pt));
    output_idx++;
    currentSifTensorIdx++;
  }

  for (const auto& pt_input_idx_and_sh_tensor :
       habana_op->GetSynImplicitOutputs()) {
    val_to_ival_map.emplace(
        node->inputs()[pt_input_idx_and_sh_tensor.pt_input_idx],
        torch::jit::IValue(
            habana_op->GetInputs()[pt_input_idx_and_sh_tensor.syn_input_idx]));
    tidx_to_tensor_map.insert(
        {currentSifTensorIdx,
         habana_op->GetInputs()[pt_input_idx_and_sh_tensor.syn_input_idx]});
    PT_DYNAMIC_SHAPE_DEBUG(
        "For implicit node output, adding to tidx_to_tensor_map: ",
        currentSifTensorIdx,
        " -> ",
        habana_helpers::DebugString(
            habana_op->GetInputs()[pt_input_idx_and_sh_tensor.syn_input_idx]));
    output_idx++;
    currentSifTensorIdx++;
  }
}

void HabanaLaunchOpPT::visit_prim_node(
    const torch::jit::Node* node,
    CValPtrtoIValueMap& val_to_ival_map) {
  if (torch::jit::prim::Constant == node->kind()) {
    for (const auto value : node->outputs()) {
      HABANA_ASSERT(val_to_ival_map.count(value) == 0);
      val_to_ival_map[value] = IVal(toIValue(value).value());
      PT_DYNAMIC_SHAPE_DEBUG(
          "For %",
          value->debugName(),
          " adding to val_to_ival_map: ",
          habana_helpers::DebugString(val_to_ival_map[value]));
      const CValPtrtoIValueMap& param_val_to_ival_map =
          jit_graph_and_meta_data_->get_param_jit_val_to_ivalue_map();
      if (param_val_to_ival_map.count(value)) {
        val_to_ival_map[value] = param_val_to_ival_map.at(value);
        PT_DYNAMIC_SHAPE_DEBUG(
            "For %",
            value->debugName(),
            " updating to val_to_ival_map: ",
            habana_helpers::DebugString(val_to_ival_map[value]),
            " w.r.t. node params");
      }
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
    auto value{node_outputs[0]};
    val_to_ival_map[value] = IVal(tensorList);
    PT_DYNAMIC_SHAPE_DEBUG(
        "For %",
        value->debugName(),
        " adding to val_to_ival_map: ",
        habana_helpers::DebugString(val_to_ival_map[value]));
  }
}

namespace {
void mapOutputTensors(
    const torch::jit::Node* node,
    const HabanaOperatorPtr& habana_op,
    const InferOutputMetaRetType& output_shape_info,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  auto op_backend = std::dynamic_pointer_cast<OpBackend>(habana_op);
  size_t nr_of_excluded_outputs = 0;
  if (op_backend) {
    nr_of_excluded_outputs = op_backend->GetSynImplicitOutputs().size();
  }

  auto nr_of_node_outputs = node->outputs().size();
  auto output_tensors = output_shape_info.GetOutputTensor();
  if (not(nr_of_node_outputs ==
          (output_tensors.size() +
           output_shape_info.GetNumUndefinedOutputTensors() -
           nr_of_excluded_outputs)) and
      output_shape_info.GetKernelOutputs().size()) {
    output_tensors =
        output_shape_info.GetKernelOutputs().at(0)->GetOutputTensor();
  }

  TORCH_CHECK(
      nr_of_node_outputs ==
          (output_tensors.size() +
           output_shape_info.GetNumUndefinedOutputTensors() -
           nr_of_excluded_outputs),
      "Output size mismatch");

  int output_iter = 0;
  for (size_t i = 0; i < nr_of_node_outputs; ++i) {
    auto output = node->outputs().at(i);
    HABANA_ASSERT(val_to_ival_map.count(output) == 0);
    if (op_backend) {
      if (not op_backend->GetOutputMetaData()[i].undefined) {
        val_to_ival_map[output] =
            IVal(std::get<1>(output_tensors[output_iter++]));
      }
    } else {
      val_to_ival_map[output] =
          IVal(std::get<1>(output_tensors[output_iter++]));
    }
  }
}

auto propagateShape(
    torch::jit::Node* node,
    torch::jit::Stack& op_input_stack,
    const HabanaOperatorPtr& habana_op,
    synapse_helpers::graph& syn_graph,
    OutputMetaDataVector& outputs_metadata,
    std::unordered_map<CValPtr, torch::jit::IValue>& val_to_ival_map) {
  PT_BRIDGE_BEGIN;

  // Create the synapse inputs from aten tensors
  create_synapse_inputs(node, habana_op, syn_graph, val_to_ival_map);

  habana_op->AllocateAndAddSynapseNode(
      syn_graph, op_input_stack, outputs_metadata);

  // process outputs
  auto output_nodes = node->outputs();

  if (output_nodes.at(0)->type() == torch::ListType::ofTensors() &&
      output_nodes.size() == 1) {
    auto unpack_node =
        jitgraph_utils::GetUnpackNodeFromTensorList(node->output(0));
    HABANA_ASSERT(
        unpack_node != nullptr,
        "TensorList is not input to ListUnpack node. Node: ",
        node->kind().toQualString());
    output_nodes = unpack_node->outputs();
  }

  size_t output_idx = 0;
  for (auto& out_tensor_pt : habana_op->GetOutputs()) {
    val_to_ival_map.emplace(
        output_nodes[output_idx], torch::jit::IValue(out_tensor_pt));
    output_idx++;
  }

  for (const auto& pt_input_idx_and_sh_tensor :
       habana_op->GetSynImplicitOutputs()) {
    val_to_ival_map.emplace(
        node->inputs()[pt_input_idx_and_sh_tensor.pt_input_idx],
        torch::jit::IValue(
            habana_op->GetInputs()[pt_input_idx_and_sh_tensor.syn_input_idx]));
  }

  PT_BRIDGE_END;
}
} // namespace

void HabanaLaunchOpPT::RunHybridSif(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& inputs,
    CValPtrtoIValueMap& val_to_ival_map) {
  using namespace sif_utils;

  PT_BRIDGE_BEGIN;

  TORCH_CHECK(inputs.size() == graph->inputs().size(), "Inputs size mismatch");

  PT_DYNAMIC_SHAPE_DEBUG(
      "SIF JIT_IR_Graph_BEGIN\n", graph->toString(), "JIT_IR_Graph_END\n");

  const auto& device = HPURegistrar::get_device();

  auto syn_graph =
      habana_helpers::create_graph(device.id(), "syn_sif_graph", true);

  mapGraphInputsToInputsOnStack(graph, inputs, val_to_ival_map);

  for (auto node : graph->nodes()) {
    std::string op_name(node->kind().toQualString());

    PT_DYNAMIC_SHAPE_DEBUG(" Visiting op ", op_name, " for node ", *node);

    // There should not be any meta ops
    HABANA_ASSERT(
        HabanaMetaOpList::isHabanaMetaOp(node->kind().toQualString()) == false,
        "Can not process meta op");

    PT_DYNAMIC_SHAPE_DEBUG(" non constant ", op_name, " found");

    // Prim nodes require special handling and are a special case
    if (node->kind().is_prim()) {
      jitgraph_utils::visit_prim_node(node, val_to_ival_map);
      continue;
    }

    auto node_type = getNodeScalarTypeFromInputs(node, val_to_ival_map);

    // Get kernel context
    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr habana_op =
        KernelRegistry().get(device.id(), op, node_type);

    TORCH_CHECK(habana_op, op, " isn't registered in KernelRegistry!");

    // Set the deterministic val
    habana_op->setDeterministic(node->i(torch::jit::attr::deterministic));

    auto op_input_stack = createInputStackForNode(node, val_to_ival_map);

    // Setup the config params for the kernels
    auto outputs_metadata = populate_node_output_metadata(node);

    if (not disabled_jit_ir_ops().count(op_name)) {
      // Set output meta data if auto-gen op
      if (auto op = std::dynamic_pointer_cast<OpBackend>(habana_op)) {
        op->SetOutputMetadata(outputs_metadata);
      }
      auto output_shape_info = habana_op->InferOutputMeta(op_input_stack);
      if (not output_shape_info.empty()) {
        // Output shape info based flow
        try {
          mapOutputTensors(node, habana_op, output_shape_info, val_to_ival_map);
        } catch (std::exception& e) {
          PT_DYNAMIC_SHAPE_DEBUG("Catch Exception SIF failed: ", e.what());
          disabled_jit_ir_ops().insert(op_name);
          propagateShape(
              node,
              op_input_stack,
              habana_op,
              syn_graph,
              outputs_metadata,
              val_to_ival_map);
        }
        continue;
      }
    }

    propagateShape(
        node,
        op_input_stack,
        habana_op,
        syn_graph,
        outputs_metadata,
        val_to_ival_map);
  }
  PT_BRIDGE_END;
}

// To instantiate the template method(s) RunHybridSif
template bool HabanaLaunchOpPT::RunHybridSif<true>(
    std::unordered_map<int64_t, at::Tensor>&,
    std::shared_ptr<std::vector<InferNodeParams>>);

template bool HabanaLaunchOpPT::RunHybridSif<false>(
    std::unordered_map<int64_t, at::Tensor>&,
    std::shared_ptr<std::vector<InferNodeParams>>);
// --------------------

// RunHybridSIF updated to return true if shape tensor(s) in the compound op(s)
// For lazy flow/compile flow:
// shape tensors are added for dynamic shapes for enabling GC shape inference.
// For eager mode:
// Shape tensors are neither supported nor added at PT bridge but for shapeless
// caching for synapse graph i.e. eager shape agnostic flow it may rely on
// GC shape inference for compound op(s), Idea is to detect such op(s) in the
// eager graph for which synapse required shape tensors.
// Such method will be only called during Eager shape agnostic cache miss i.e.
// non-critical path with RunHybridSif<DynamicShapes == true>().
template <bool DynamicShapes>
bool HabanaLaunchOpPT::RunHybridSif(
    std::unordered_map<int64_t, at::Tensor>& tidx_to_tensor_map,
    std::shared_ptr<std::vector<InferNodeParams>> node_params_vec_ptr) {
  PT_BRIDGE_BEGIN;

  bool shape_tensors_flag = false;
  PT_DYNAMIC_SHAPE_DEBUG(
      "\nRunning hybrid shape inference on graph: ", GetSynapseGraphName());
  habana::PrintStack(*pt_stack);
  PT_DYNAMIC_SHAPE_DEBUG(
      "JIT_IR_Graph_BEGIN\n", jit_ir_graph_->toString(), "JIT_IR_Graph_END\n");

  CValPtrtoIValueMap val_to_ival_map;
  auto graph_inputs = jit_ir_graph_->inputs();
  TORCH_CHECK(input_refs.size() == graph_inputs.size(), "Input size mismatch");
  for (size_t i = 0; i < graph_inputs.size(); i++) {
    auto input = graph_inputs[i];
    val_to_ival_map[input] = input_refs[i];
  }

  // Figure out the right device id
  auto& device = HPURegistrar::get_device();
  synDeviceId device_id = device.id();

  auto syn_graph =
      habana_helpers::create_graph(device.id(), GetSynapseGraphName(), true);
  if constexpr (DynamicShapes) {
    syn_graph.set_dynamic_graph(true);
  }

  std::vector<at::Tensor> input_shape_tensors_vec;
  std::vector<at::Tensor> intermediate_shape_tensors_vec;

  for (auto* node : jit_ir_graph_->nodes()) {
    std::string op_name(node->kind().toQualString());

    PT_DYNAMIC_SHAPE_DEBUG(" Visiting op ", op_name, " for node ", *node);

    // There should not be any meta ops
    HABANA_ASSERT(
        HabanaMetaOpList::isHabanaMetaOp(node->kind().toQualString()) == false,
        "Can not process meta op");

    // Prim nodes require special handling and are a special case
    if (node->kind().is_prim()) {
      PT_DYNAMIC_SHAPE_DEBUG(" constant ", op_name, " found");
      visit_prim_node(node, val_to_ival_map);
      continue;
    }

    PT_DYNAMIC_SHAPE_DEBUG(" non constant ", op_name, " found");

    // TODO: visit restride nodes
    if ((strcmp(op_name.c_str(), "hpu::restride_cl") == 0) ||
        (strcmp(op_name.c_str(), "hpu::restride") == 0)) {
      PT_DYNAMIC_SHAPE_DEBUG("Restride found, skipping ...");
      continue;
    }

    // Get node scalar type, Default value Float if no tensor is found
    c10::ScalarType node_type = c10::ScalarType::Float;
    for (auto input : node->inputs()) {
      if (val_to_ival_map.count(input) && val_to_ival_map[input].isTensor()) {
        node_type = val_to_ival_map[input].toTensor().scalar_type();
        break;
      }
    }

    // Get kernel context
    const auto& op = node->schema().operator_name();
    HabanaOperatorPtr habana_op =
        KernelRegistry().get(device_id, op, node_type);

    TORCH_CHECK(habana_op, op, " isn't registered in KernelRegistry!");

    // Set the deterministic val
    habana_op->setDeterministic(node->i(torch::jit::attr::deterministic));

    // Set kernel execution mode
    habana_op->SetExecutionMode(execution_mode_);

    bool is_mapped_flag{true};
    auto op_input_stack =
        create_stack_for_node(node, is_mapped_flag, val_to_ival_map);
    HABANA_ASSERT(
        is_mapped_flag, "Cannot proceed with unmapped input for ", op_name);
    // This log line is used by the logging analysis tool. Please be cautious
    // when changing.
    PT_OP_INFO("JIT_OP SIF_OUTPUT ", OpInfo::DumpOpInfo(op, op_input_stack));

    // If there is a "meta attribute" marked with attr::arg1, add the meta attr
    // value to stack for the ops to work with. At this point, only StridedView
    // ops in eager mode uses it.
    auto meta = torch::jit::attr::arg1;
    if (node->hasAttribute(meta)) {
      HABANA_ASSERT(
          !strcmp("aten::as_strided", node->kind().toQualString()),
          "Meta op can only be marked for aten::as_strided, not supported in op ",
          node->kind().toQualString());
      op_input_stack.insert(op_input_stack.end(), IValue(node->i(meta)));
    }

    // Collect input shape tensors, To add them at last after graph inputs
    // Add only shape tensors and input describing shape tensors and
    // exclude front end shape tensors added for H2D tensors
    if constexpr (DynamicShapes) {
      for (auto const& input : op_input_stack) {
        if (input.isTensor()) {
          auto tensor = input.toTensor();
          auto tmeta{get_tensor_extra_meta(tensor, true)};
          if (tmeta && tmeta->is_H2D_frontend_shape_tensor() == false &&
              tmeta->get_tensor_type() == SHAPE_TENSOR) {
            input_shape_tensors_vec.emplace_back(tensor);
          }
        }
      }
    }

    // Setup the config params for the kernels
    auto outputs_metadata = populate_node_output_metadata(node);

    auto propagate_shape{[&]() -> void {
      PT_BRIDGE_BEGIN;
      // Non OutputShapeInf based path, adjust SifTensrorId
      PT_DYNAMIC_SHAPE_DEBUG(
          "Using non OutputShapeInf based flow. Going to add tpc kernel ",
          habana_op->GetGuid(),
          " for ",
          op_name);

      // Create the synapse inputs from aten tensors
      create_synapse_inputs(node, habana_op, syn_graph, val_to_ival_map);

      habana_op->AllocateAndAddSynapseNode(
          syn_graph, op_input_stack, outputs_metadata);

      constexpr int one = 1;
      size_t num_syn_nodes = habana_op->GetKernels().size() + one;
      if (auto op = std::dynamic_pointer_cast<OpBackend>(habana_op)) {
        num_syn_nodes = op->GetNumSynNodes();
      }

      if constexpr (DynamicShapes) {
        process_shape_tensors(habana_op, intermediate_shape_tensors_vec);
        shape_tensors_flag |=
            (intermediate_shape_tensors_vec.size() && (num_syn_nodes > one));
      }
      process_outputs(habana_op, node, val_to_ival_map, tidx_to_tensor_map);

      auto output_count = get_output_tensors_count(habana_op, syn_graph);
      habana::ShapeInference::IncrementSifTensorId(output_count);
      PT_DYNAMIC_SHAPE_DEBUG(
          "After increment: sif tensor id = ",
          habana::ShapeInference::GetSifTensorId());

      // ToDO: Add support for node params if SIF method not available
      if (node_params_vec_ptr) {
        // add dummy node params for all sub-kernels/syn nodes if any
        for (size_t i = 0; i < num_syn_nodes; i++) {
          (*node_params_vec_ptr).push_back(InferNodeParams(nullptr, 0));
        }
      }
    }};

    if (!disabled_jit_ir_ops().count(op_name)) {
      // Set output meta data if auto-gen op
      if (auto op = std::dynamic_pointer_cast<OpBackend>(habana_op)) {
        op->SetOutputMetadata(outputs_metadata);
      }
      auto output_shape_info = habana_op->InferOutputMeta(op_input_stack);
      if (output_shape_info.empty()) {
        PT_DYNAMIC_SHAPE_DEBUG("OutputShapeInf is not supported for ", op_name);
        propagate_shape();
      } else {
        // Output shape info based flow
        PT_DYNAMIC_SHAPE_DEBUG(
            "Using OutputShapeInf shape info based flow for ",
            habana_op->GetGuid(),
            ", ",
            op_name);
        auto output_tensors = output_shape_info.GetOutputTensor();

        constexpr int one = 1;
        size_t exclude_outputs = 0;
        size_t num_syn_nodes = habana_op->GetKernels().size() + one;
        if (auto op = std::dynamic_pointer_cast<OpBackend>(habana_op)) {
          exclude_outputs = op->GetSynImplicitOutputs().size();
          num_syn_nodes = op->GetNumSynNodes();
        }
        // Collect all output tensors
        for (auto& t : output_tensors) {
          auto curSifTidx{std::get<0>(t)};
          auto out_tensor_pt{std::get<1>(t)};
          PT_DYNAMIC_SHAPE_DEBUG(
              "For node output with cs, adding to tidx_to_tensor_map: ",
              curSifTidx,
              " -> ",
              habana_helpers::DebugString(out_tensor_pt));
          tidx_to_tensor_map.insert({curSifTidx, out_tensor_pt});
        }

        // Recursivly collect all shape tensors
        if constexpr (DynamicShapes) {
          std::vector<IdxTensorTuple> intermediate_shape_tensor_cs;
          ProcessShapeTensorsCS(
              output_shape_info, intermediate_shape_tensor_cs);
          shape_tensors_flag |=
              (intermediate_shape_tensor_cs.size() && (num_syn_nodes > one));

          // Get all values of shape tensor
          for (auto& t : intermediate_shape_tensor_cs) {
            auto curSifTidx{std::get<0>(t)};
            auto shape_tensor_pt{std::get<1>(t)};
            PT_DYNAMIC_SHAPE_DEBUG(
                "For node shape output with cs, adding to tidx_to_tensor_map: ",
                curSifTidx,
                " -> ",
                habana_helpers::DebugString(shape_tensor_pt));
            tidx_to_tensor_map.insert({curSifTidx, shape_tensor_pt});
          }
        }

        HABANA_ASSERT(
            node->outputs().size() == output_tensors.size() - exclude_outputs);
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          auto output = node->outputs().at(i);
          HABANA_ASSERT(val_to_ival_map.count(output) == 0);
          val_to_ival_map[output] = IVal(std::get<1>(output_tensors[i]));
        }

        // Capture node params if required and are supported per JIT IR op
        if (node_params_vec_ptr) {
          const auto& params = output_shape_info.GetNodeParams();
          // node params patching supported
          if (NodeParamAgnosticOpList::isNodeParamAgnosticOp(
                  c10::Symbol::fromQualString(op_name))) {
            for (const auto& p : params) {
              PT_DYNAMIC_SHAPE_DEBUG(
                  "For node params with cs, adding params data: ",
                  p.get_data(),
                  ", params size: ",
                  p.get_size());
              (*node_params_vec_ptr).push_back(p);
            }
          } else { // node params patching not supported
            // add dummy params for all syn nodes/sub-kernels
            size_t nodesCount = !params.empty()
                ? params.size()
                : output_shape_info.GetKernels().size() + 1;
            for (size_t i = 0; i < nodesCount; i++) {
              (*node_params_vec_ptr).push_back(InferNodeParams(nullptr, 0));
            }
          }
        }
      }
    } else {
      PT_DYNAMIC_SHAPE_DEBUG("OutputShapeInf is disabled for ", op_name);
      propagate_shape();
    }
  }

  if constexpr (DynamicShapes) {
    // For all Graph inputs create a sif mapping
    for (size_t i = 0; i < graph_inputs.size(); ++i) {
      if (input_refs[i].isScalar())
        continue;
      HABANA_ASSERT(input_refs[i].isTensor());
      auto inp_sif_tid = habana::ShapeInference::ReadAndIncrementSifTensorId();
      tidx_to_tensor_map.insert({inp_sif_tid, input_refs[i].toTensor()});
      PT_DYNAMIC_SHAPE_DEBUG(
          "For graph inputs, adding to tidx_to_tensor_map: ",
          inp_sif_tid,
          " -> ",
          habana_helpers::DebugString(input_refs[i].toTensor()));
    }

    // For all input shape tensors create a sif mapping
    for (auto const& input_tensor : input_shape_tensors_vec) {
      auto inp_sif_tid = habana::ShapeInference::ReadAndIncrementSifTensorId();
      tidx_to_tensor_map.insert({inp_sif_tid, input_tensor});
      PT_DYNAMIC_SHAPE_DEBUG(
          "For graph shape inputs, adding to tidx_to_tensor_map: ",
          inp_sif_tid,
          " -> ",
          habana_helpers::DebugString(input_tensor));
    }

    // For all intermediate shape tensors for nodes not supporting
    // OutputShapeInf create a sif mapping
    for (auto const& inter_tensor : intermediate_shape_tensors_vec) {
      auto inter_sif_tid =
          habana::ShapeInference::ReadAndIncrementSifTensorId();
      tidx_to_tensor_map.insert({inter_sif_tid, inter_tensor});
      PT_DYNAMIC_SHAPE_DEBUG(
          "For graph intermediate shape tensors, adding to tidx_to_tensor_map: ",
          inter_sif_tid,
          " -> ",
          habana_helpers::DebugString(inter_tensor));
    }
  }

  PT_BRIDGE_END;
  return shape_tensors_flag;
}

} // namespace habana
