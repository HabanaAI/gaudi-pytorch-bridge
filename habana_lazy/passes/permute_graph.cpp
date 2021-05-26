/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "permute_graph.h"
#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/lazy_kernels_declarations.h"
using namespace torch::jit;
namespace habana_lazy {

bool IsNodeLayoutAgnostic(const Node* node) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  habana::HabanaOperatorPtr habana_kernel = habana::KernelRegistry().get(
      device_id, node->kind().toQualString(), c10::ScalarType::Float);
  auto& habana_kernel_meta_data = habana_kernel->GetKernelMetaData();
  auto node_ins = node->inputs();

  size_t tensor_idx = 0;
  habana::LayoutFormat config_layout = habana::LayoutFormat::ANY;
  size_t meta_size = habana_kernel_meta_data.input_layout.size();
  for (const auto value_in : node_ins) {
    if (value_in->type()->kind() == c10::TypeKind::TensorType) {
      config_layout = tensor_idx >= meta_size
          ? habana::LayoutFormat::ANY
          : habana_kernel_meta_data.input_layout.at(tensor_idx);
      if (config_layout != habana::LayoutFormat::ANY)
        return false;
    }
    tensor_idx++;
  }

  size_t output_tensor_idx = 0;
  auto node_outs = node->outputs();
  meta_size = habana_kernel_meta_data.output_layout.size();
  habana::LayoutFormat out_layout = habana::LayoutFormat::ANY;
  for (const auto value_out : node_outs) {
    if (value_out->type()->kind() == c10::TypeKind::TensorType) {
      out_layout = output_tensor_idx >= meta_size
          ? habana::LayoutFormat::ANY
          : habana_kernel_meta_data.output_layout.at(output_tensor_idx);
      if (out_layout != habana::LayoutFormat::ANY)
        return false;
    }
    output_tensor_idx++;
  }
  return true;
}

bool WeightIdentificationPass::isTensor(const torch::jit::Value* value) {
  HABANA_ASSERT(value->node());
  return !(value->node()->kind() == c10::prim::Constant);
}

void WeightIdentificationPass::markInputs(const torch::jit::Value* in) {
  auto node = in->node();
  if (nullptr == node) {
    return;
  }
  std::string node_str = node->kind().toQualString();
  if (0 == kernelWeightIdx.count(node_str)) {
    for (auto& i : node->inputs()) {
      if (isTensor(i) && !weightTensors.count(i)) {
        weightTensors.insert(i);
        markInputs(i);
      }
    }
  }
}

void WeightIdentificationPass::markOutputs(const torch::jit::Value* in) {
  for (auto& use : in->uses()) {
    auto node = use.user;
    HABANA_ASSERT(node);

    // TODO: check if its binary/unary ops & then mark
    std::string node_str = node->kind().toQualString();
    if (0 == kernelWeightIdx.count(node_str)) {
      for (auto& out : node->outputs()) {
        if (!weightTensors.count(out)) {
          weightTensors.insert(out);
          markOutputs(out);
        }
      }
    }
  }
}

void WeightIdentificationPass::markWeights(const torch::jit::Value* in) {
  markInputs(in);
  markOutputs(in);
}

void WeightIdentificationPass::markWeightTensors(
    std::shared_ptr<Graph>& graph) {
  for (auto node : graph->nodes()) {
    std::string kernel = node->kind().toQualString();
    auto it = kernelWeightIdx.find(kernel);
    if (kernelWeightIdx.end() != it) {
      auto weightIdx = it->second;
      HABANA_ASSERT(weightIdx < node->inputs().size());
      auto weightIn = node->inputs()[weightIdx];
      weightTensors.insert(weightIn);
      markWeights(weightIn);
    }
  }
}

at::IntArrayRef getDimsForLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order) {
  at::IntArrayRef dims;

  if (current_order == habana::LayoutFormat::NCHW) {
    if (channel_order == habana::LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {0, 2, 3, 1};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {2, 3, 1, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::NHWC) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {0, 3, 1, 2};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {1, 2, 3, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::HWCK) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {3, 2, 0, 1};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {3, 0, 1, 2};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermtue_graph: permute called for unsupported channel order");
    }
  } else {
    TORCH_CHECK(
        0,
        " InsertPermtue_graph: permute called for unsupported channel order");
  }

  return dims;
}

bool isRetunrOut(std::shared_ptr<Graph>& graph, const Value* value_out) {
  auto node_return = graph->return_node();
  for (auto value : node_return->inputs()) {
    if (value_out == value)
      return true;
  }
  return false;
}

using ValuePtrTensorLayoutMap =
    std::unordered_map<const torch::jit::Value*, habanaTensorLayoutInfo>;
using NodePtrVecIndxDimsMap = std::unordered_map<
    torch::jit::Node*,
    std::vector<std::pair<torch::jit::Value*, at::IntArrayRef>>>;

bool IsPermuteNode(const Node* node) {
  return (
      (strcmp(node->kind().toQualString(), "hpu::permute_cl") == 0) ||
      (strcmp(node->kind().toQualString(), "aten::permute") == 0));
}

void InsertNodes(
    std::shared_ptr<Graph>& graph,
    const std::string& op,
    NodePtrVecIndxDimsMap node_indxmap) {
  // insert nodes in graph
  for (auto node_map : node_indxmap) {
    auto anchor_node = node_map.first;
    for (auto offset_layout : node_map.second) {
      auto value = offset_layout.first;
      auto dims = offset_layout.second;
      WithInsertPoint insert_point(anchor_node);
      auto op_permute = c10::Symbol::fromQualString(op);
      auto value_dims = graph->insertConstant(IValue(dims));
      auto permute_node = graph->create(op_permute, {value, value_dims}, 1);
      graph->insertNode(permute_node);
      anchor_node->replaceInputWith(value, permute_node->output(0));
    }
  }
}

void InsertPermuteNodes(
    std::shared_ptr<Graph>& graph,
    NodePtrVecIndxDimsMap node_indxmap) {
  InsertNodes(graph, "aten::permute", node_indxmap);
}
void InsertRestrideNodes(
    std::shared_ptr<Graph>& graph,
    NodePtrVecIndxDimsMap node_indxmap) {
  InsertNodes(graph, "hpu::restride_cl", node_indxmap);
}

void RemoveRedundantOp(std::shared_ptr<Graph>& graph, const char* op) {
  torch::jit::graph_node_list graph_nodes = graph->nodes();
  for (auto* node : graph_nodes) {
    if (strcmp(node->kind().toQualString(), op) == 0) {
      auto uses = node->input(0)->uses();
      for (auto u : uses) {
        auto perm_node = u.user;
        if (perm_node != node &&
            strcmp(perm_node->kind().toQualString(), op) == 0) {
          auto perm_dims = toIValue(perm_node->input(1))->toIntVector();
          auto node_dims = toIValue(node->input(1))->toIntVector();
          if (node_dims == perm_dims) {
            WithInsertPoint insert_point(perm_node);
            perm_node->output(0)->replaceAllUsesWith(node->output(0));
            perm_node->destroy();
          }
        }
      }
    }
  }
}

void RemoveRedundantPermutes(std::shared_ptr<Graph>& graph) {
  RemoveRedundantOp(graph, "aten::permute");
}
void RemoveRedundantRestrideNodes(std::shared_ptr<Graph>& graph) {
  RemoveRedundantOp(graph, "hpu::restride_cl");
}

/*Layout optimization pass
  1. Parse each node inputs and add permutes only for layout non-agnostic nodes
  2. ChannelsLast nodes should return restrided output since PT expects
     format in except in NCHW format.
  3. Node layout info is passed to output Value
     a. Assign layout format as per kernel meta data
     b. Single input/output nodes should pass input layout info
     c. Existing permute nodes in the graph should pass dims Layout info to
  output d. Non layout agnostic nodes pass first input layout info this is the
  assumption and in most cases this should be sufficient and any corner cases
  should be added as special cases like hpu::cast, hpu::habana_d2d_memcpy_other
  etc
  4. Graph return outputs should add permutes if entry and config layout
  mismatch
  5. Graph return outputs add restride node if inputs are ChannelsLast
  6. Finally remove duplicate permutes in the graph
*/

void InsertPermute_graph(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack) {
  auto graph_inputs = graph->inputs();
  ValuePtrTensorLayoutMap value_to_tensor_layout;
  size_t idx = stack.size() - graph->inputs().size();
  for (size_t j = 0; j < graph_inputs.size(); j++) {
    auto value_input = graph_inputs[j];
    if (stack[idx + j].isTensor()) {
      auto tensor = stack[idx + j].toTensor();
      if (tensor.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
          tensor.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d) {
        value_to_tensor_layout[value_input].layout = habana::LayoutFormat::NHWC;
        value_to_tensor_layout[value_input].layout_at_graph_entry =
            habana::LayoutFormat::NHWC;
        // Add restride nodes to all inputs
        auto node_insert = value_input->uses().at(0).user;
        WithInsertPoint insert_point(node_insert);
        auto op_restride = c10::Symbol::fromQualString("hpu::restride_cl");
        auto dims = getDimsForLayout(
            habana::LayoutFormat::NHWC, habana::LayoutFormat::NCHW);
        auto value_dims = graph->insertConstant(IValue(dims));
        auto restride_node =
            graph->create(op_restride, {value_input, value_dims}, 1);
        graph->insertNode(restride_node);
        value_input->replaceAllUsesAfterNodeWith(
            restride_node, restride_node->output(0));
        // mark output layouts
        value_to_tensor_layout[restride_node->output(0)].layout =
            habana::LayoutFormat::NHWC;
        value_to_tensor_layout[restride_node->output(0)].layout_at_graph_entry =
            habana::LayoutFormat::NHWC;
      } else {
        value_to_tensor_layout[value_input].layout = habana::LayoutFormat::NCHW;
        value_to_tensor_layout[value_input].layout_at_graph_entry =
            habana::LayoutFormat::NCHW;
      }
    }
  }
  WeightIdentificationPass weight_pass;
  weight_pass.markWeightTensors(graph);

  auto weight_values = weight_pass.getWeightTensors();
  for (auto win : weight_values) {
    value_to_tensor_layout[win].layout = habana::LayoutFormat::HWCK;
    value_to_tensor_layout[win].layout_at_graph_entry =
        habana::LayoutFormat::HWCK;
  }

  std::unordered_map<
      torch::jit::Node*,
      std::vector<std::pair<torch::jit::Value*, at::IntArrayRef>>>
      anchor_nodes_;
  std::unordered_map<
      torch::jit::Node*,
      std::vector<std::pair<torch::jit::Value*, at::IntArrayRef>>>
      anchor_restride_nodes_;
  torch::jit::graph_node_list graph_nodes = graph->nodes();

  for (auto* node : graph_nodes) {
    if (node == graph->param_node() ||
        node->kind() == torch::jit::prim::Constant ||
        node->kind() == torch::jit::prim::ListConstruct ||
        node->kind() == torch::jit::prim::dtype ||
        (strcmp(node->kind().toQualString(), "hpu::restride_cl") == 0))
      continue;

    // ListUnpack tensors should set layout format
    if (node->kind() == torch::jit::prim::ListUnpack) {
      for (auto value_out : node->outputs()) {
        if (value_out->type()->kind() == c10::TypeKind::TensorType) {
          value_to_tensor_layout[value_out].layout = habana::LayoutFormat::NCHW;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              habana::LayoutFormat::NCHW;
        }
      }
      continue;
    }

    // Get kernel MetaData
    auto& device = synapse_helpers::HPURegistrar::get_device();
    synDeviceId device_id = device.id();
    habana::HabanaOperatorPtr habana_kernel = habana::KernelRegistry().get(
        device_id, node->kind().toQualString(), c10::ScalarType::Float);
    auto& habana_kernel_meta_data = habana_kernel->GetKernelMetaData();
    bool isLayoutAgnostic = IsNodeLayoutAgnostic(node);

    // permute Loop
    auto node_ins = node->inputs();
    size_t tensor_idx = 0;
    habana::LayoutFormat in_layout, prev_layout = habana::LayoutFormat::ANY;
    size_t meta_size = habana_kernel_meta_data.input_layout.size();
    for (const auto value_in : node_ins) {
      if (value_in->type()->kind() == c10::TypeKind::TensorType) {
        in_layout = tensor_idx >= meta_size
            ? habana::LayoutFormat::ANY
            : habana_kernel_meta_data.input_layout.at(tensor_idx);

        TORCH_CHECK(
            value_to_tensor_layout.find(value_in) !=
                std::end(value_to_tensor_layout),
            "InsertPermtue_graph : Channel order not updated");
        auto tensor_layout = value_to_tensor_layout[value_in].layout;

        // For weight tensors we update the map before execution starts
        // through weightmarking pass If its marked HWCK in the map,
        // we can override with it
        if (tensor_layout == habana::LayoutFormat::HWCK) {
          TORCH_CHECK(
              in_layout == habana::LayoutFormat::HWCK ||
                  in_layout == habana::LayoutFormat::ANY,
              "InsertPermute_graph, got contradicting layout info from meta data and opt pass");
          in_layout = habana::LayoutFormat::HWCK;
        }

        if (in_layout == habana::LayoutFormat::ANY && tensor_idx > 0) {
          in_layout = prev_layout;
        }

        if (in_layout == habana::LayoutFormat::HWCK) {
          in_layout = habana::LayoutFormat::ANY;
          value_to_tensor_layout[value_in].layout = habana::LayoutFormat::HWCK;
        }

        bool permute_required =
            (in_layout != tensor_layout &&
             in_layout != habana::LayoutFormat::ANY);
        auto perm_layout = in_layout;
        if ((strcmp(node->kind().toQualString(), "aten::view") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::index") == 0)) {
          auto tensor_entry_layout =
              value_to_tensor_layout[value_in].layout_at_graph_entry;
          if ((tensor_layout != tensor_entry_layout) &&
              tensor_layout != habana::LayoutFormat::HWCK) {
            permute_required = true;
            perm_layout = tensor_entry_layout;
          }
        }

        // add permtues in the graph
        if (permute_required) {
          if (*value_in->type()->cast<TensorType>()->dim() == 4) {
            auto dims = getDimsForLayout(perm_layout, tensor_layout);
            anchor_nodes_[node].push_back(std::make_pair(value_in, dims));
            tensor_layout = perm_layout;
          }
        }
        prev_layout = tensor_idx == 0 ? tensor_layout : prev_layout;
        tensor_idx++;
      }
    }

    // Pass layout info to Node outputs
    if (!isLayoutAgnostic) {
      // Nodes with meta layout info
      size_t output_tensor_idx = 0;
      auto node_outs = node->outputs();
      auto in_layout_entry =
          value_to_tensor_layout[node->input(0)].layout_at_graph_entry;
      if (in_layout_entry == habana::LayoutFormat::ANY) {
        in_layout_entry = habana::LayoutFormat::NCHW;
      }
      for (const auto value_out : node_outs) {
        auto out_layout =
            habana_kernel_meta_data.output_layout.at(output_tensor_idx);
        if (out_layout == habana::LayoutFormat::ANY) {
          value_to_tensor_layout[value_out].layout = habana::LayoutFormat::NCHW;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              habana::LayoutFormat::NCHW;
        } else {
          value_to_tensor_layout[value_out].layout = out_layout;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              in_layout_entry;
          if (out_layout == habana::LayoutFormat::HWCK) {
            value_to_tensor_layout[value_out].layout_at_graph_entry =
                out_layout;
          }
        }
        output_tensor_idx++;
      }
    } else {
      // Node has single input and output
      if ((node->outputs().size() == 1) && (node->inputs().size() == 1)) {
        size_t out_meta_size = habana_kernel_meta_data.output_layout.size();
        habana::LayoutFormat out_layout = habana::LayoutFormat::ANY;
        auto value_in = node->input(0);
        auto value_out = node->output(0);
        if (out_meta_size == 0) {
          out_layout = habana::LayoutFormat::ANY;
        } else {
          out_layout = habana_kernel_meta_data.input_layout.at(0);
        }
        if (out_layout == habana::LayoutFormat::ANY) {
          value_to_tensor_layout[value_out].layout =
              value_to_tensor_layout[value_in].layout;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              value_to_tensor_layout[value_in].layout_at_graph_entry;
        } else {
          value_to_tensor_layout[value_out].layout = out_layout;
        }
      } else if (
          // permute_cl sets channelsLast output format
          (strcmp(node->kind().toQualString(), "hpu::permute_cl") == 0)) {
        auto value_out = node->output(0);
        value_to_tensor_layout[value_out].layout = habana::LayoutFormat::NHWC;
        value_to_tensor_layout[value_out].layout_at_graph_entry =
            habana::LayoutFormat::NHWC;
      } else if (
          // special case format info passing
          (strcmp(node->kind().toQualString(), "hpu::cast") == 0) ||
          (strcmp(
               node->kind().toQualString(), "hpu::habana_d2d_memcpy_other") ==
           0)) {
        auto value_in = node->input(0);
        auto in_layout = value_to_tensor_layout[value_in].layout;
        auto in_layout_entry =
            value_to_tensor_layout[value_in].layout_at_graph_entry;
        for (auto value_out : node->outputs()) {
          value_to_tensor_layout[value_out].layout = in_layout;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              in_layout_entry;
        }
      } else if (
          (strcmp(node->kind().toQualString(), "aten::view") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::index") == 0)) {
        // View() layout is always NCHW as per original PT format
        // [ToDo] consider case permute_cl followed by view()
        // %1 = aten::permute_cl(...)
        // %2 = aten::view(%1)
        auto value_out = node->output(0);
        auto value_in = node->input(0);
        value_to_tensor_layout[value_out].layout = habana::LayoutFormat::NCHW;
        value_to_tensor_layout[value_out].layout_at_graph_entry =
            value_to_tensor_layout[value_in].layout_at_graph_entry;
      } else {
        // Multi input and single output pass layout info from input to output
        auto node_outs = node->outputs();
        meta_size = habana_kernel_meta_data.output_layout.size();
        size_t output_tensor_idx = 0, node_idx = 0;
        habana::LayoutFormat assigned_input_layout = habana::LayoutFormat::NCHW;
        habana::LayoutFormat origin_input_layout = habana::LayoutFormat::NCHW;
        for (const auto value_in : node_ins) {
          if (value_in->type()->kind() == c10::TypeKind::TensorType) {
            TORCH_CHECK(
                value_to_tensor_layout.find(value_in) !=
                    std::end(value_to_tensor_layout),
                "InsertPermtue_graph : Channel order not updated");
            auto in_layout = value_to_tensor_layout[value_in].layout;
            assigned_input_layout =
                ((node_idx == 0) || in_layout == habana::LayoutFormat::HWCK)
                ? in_layout
                : assigned_input_layout;

            origin_input_layout =
                value_to_tensor_layout[value_in].layout_at_graph_entry ==
                    habana::LayoutFormat::NHWC
                ? habana::LayoutFormat::NHWC
                : origin_input_layout;
            node_idx++;
          }
        }

        for (const auto value_out : node_outs) {
          value_to_tensor_layout[value_out].layout = assigned_input_layout;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              origin_input_layout;
          if (assigned_input_layout == habana::LayoutFormat::HWCK) {
            value_to_tensor_layout[value_out].layout = assigned_input_layout;
            value_to_tensor_layout[value_out].layout_at_graph_entry =
                assigned_input_layout;
          }
          output_tensor_idx++;
        }
      }
    }
  }

  // permutes on return as per original PT layout
  auto node_return = graph->return_node();
  size_t ret_idx = 0;
  for (auto value_out : node_return->inputs()) {
    auto prev_layout = value_to_tensor_layout[value_out].layout_at_graph_entry;
    auto cur_layout = value_to_tensor_layout[value_out].layout;
    if (value_out->type()->kind() == c10::TypeKind::TensorType) {
      if (prev_layout == habana::LayoutFormat::NHWC) {
        if (cur_layout == habana::LayoutFormat::NHWC) {
          std::string node_str = value_out->node()->kind().toQualString();
          if (node_str == "aten::select") {
            at::IntArrayRef dims;
            static const int64_t dimarr[] = {2, 0, 1};
            dims = dimarr;
            anchor_restride_nodes_[node_return].push_back(
                std::make_pair(value_out, dims));
          } else {
            at::IntArrayRef dims;
            static const int64_t dimarr[] = {0, 3, 1, 2};
            dims = dimarr;
            anchor_restride_nodes_[node_return].push_back(
                std::make_pair(value_out, dims));
          }
        }
      } else {
        if (prev_layout != cur_layout) {
          if (*value_out->type()->cast<TensorType>()->dim() == 4) {
            auto dims = getDimsForLayout(prev_layout, cur_layout);
            anchor_nodes_[node_return].push_back(
                std::make_pair(value_out, dims));
          }
        }
      }
    }
    ret_idx++;
  }

  // insert permute nodes and Restride nodes in graph
  InsertPermuteNodes(graph, anchor_nodes_);
  InsertRestrideNodes(graph, anchor_restride_nodes_);

  // Remove redundant/duplicate permutes and restride nodes
  // [toDo] Avoid RemoveRedundantPermutes calls.
  // redundant nodes are created as per the following graph
  // and repalceAllUses of %1 while parsing each node
  // causing issues. That is why added a seperate pass to remove duplicate
  // permute nodes.
  //
  // %1 = permute(%0)
  // conv(%1, ...)
  // :
  // %2 = permute(%0)
  // conv_backward(%2)
  RemoveRedundantPermutes(graph);
  RemoveRedundantRestrideNodes(graph);
}
} // namespace habana_lazy
