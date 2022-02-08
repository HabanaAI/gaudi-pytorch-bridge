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
#include "pass_utils.h"
using namespace torch::jit;
namespace habana_lazy {

bool IsNodeLayoutAgnostic(const Node* node) {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  habana::HabanaOperatorPtr habana_kernel = habana::KernelRegistry().get(
      device_id, node->schema().operator_name(), c10::ScalarType::Float);
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

at::IntArrayRef getDimsForLayout5d(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order) {
  at::IntArrayRef dims;

  // using NCHW/NHWC/HWCK since synapse 5d layout nomenclature
  // is not clear. Note that for 5d layout channel dim is 1 for
  // NCDHW and 4 for NDHWC
  if (current_order == habana::LayoutFormat::NCHW) {
    if (channel_order == habana::LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {0, 2, 3, 4, 1};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {2, 3, 4, 1, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermute_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::NHWC) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {0, 4, 1, 2, 3};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::HWCK) {
      static const int64_t dimarr[] = {1, 2, 3, 4, 0};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermute_graph: permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::HWCK) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      static const int64_t dimarr[] = {4, 3, 0, 1, 2};
      dims = dimarr;
    } else if (channel_order == habana::LayoutFormat::NHWC) {
      static const int64_t dimarr[] = {4, 0, 1, 2, 3};
      dims = dimarr;
    } else {
      TORCH_CHECK(
          0,
          " InsertPermute_graph: permute called for unsupported channel order");
    }
  } else {
    TORCH_CHECK(
        0,
        " InsertPermute_graph: permute called for unsupported channel order");
  }

  return dims;
}

using ValuePtrTensorLayoutMap =
    std::unordered_map<const torch::jit::Value*, habanaTensorLayoutInfo>;
using NodePtrVecIndxDimsMap = std::unordered_map<
    torch::jit::Node*,
    std::vector<std::pair<torch::jit::Value*, at::IntArrayRef>>>;

bool IsPermuteNode(const Node* node) {
  return (
      (strcmp(node->kind().toQualString(), "hpu::permute_cl") == 0) ||
      (strcmp(node->kind().toQualString(), "hpu::permute") == 0));
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
  InsertNodes(graph, "hpu::permute", node_indxmap);
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
  RemoveRedundantOp(graph, "hpu::permute");
}
void RemoveRedundantRestrideNodes(std::shared_ptr<Graph>& graph) {
  RemoveRedundantOp(graph, "hpu::restride_cl");
}

static const std::unordered_map<std::string, size_t> dimBasedOpsIdx = {
    {"aten::slice", 1}};

bool isDimBasedOp(const Node* node) {
  return node ? dimBasedOpsIdx.count(node->kind().toQualString()) != 0 : false;
}

int64_t getLayoutDim5d(habana::LayoutFormat layout, int64_t dim) {
  // using NCHW/NHWC/HWCK since synapse 5d layout nomenclature
  // is not clear. Note that for 5d layout channel dim is 1 for
  // NCDHW and 4 for NDHWC
  int layout_dim = dim;
  if (layout == habana::LayoutFormat::NCHW) {
    int64_t dimarr[] = {0, 1, 2, 3, 4};
    layout_dim = dimarr[dim];
  } else if (layout == habana::LayoutFormat::NHWC) {
    int64_t dimarr[] = {0, 4, 1, 2, 3};
    layout_dim = dimarr[dim];
  } else if (layout == habana::LayoutFormat::HWCK) {
    int64_t dimarr[] = {4, 3, 0, 1, 2};
    layout_dim = dimarr[dim];
  } else {
    HABANA_ASSERT(0);
  }
  return layout_dim;
}

int64_t getLayoutDim(habana::LayoutFormat layout, int64_t dim) {
  int layout_dim = dim;
  if (layout == habana::LayoutFormat::NCHW) {
    int64_t dimarr[] = {0, 1, 2, 3};
    layout_dim = dimarr[dim];
  } else if (layout == habana::LayoutFormat::NHWC) {
    int64_t dimarr[] = {0, 3, 1, 2};
    layout_dim = dimarr[dim];
  } else if (layout == habana::LayoutFormat::HWCK) {
    int64_t dimarr[] = {3, 2, 0, 1};
    layout_dim = dimarr[dim];
  } else {
    HABANA_ASSERT(0);
  }
  return layout_dim;
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
      auto is_5d_layout =
          tensor.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
      if (tensor.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
          tensor.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d) {
        value_to_tensor_layout[value_input].layout = habana::LayoutFormat::NHWC;
        value_to_tensor_layout[value_input].layout_at_graph_entry =
            habana::LayoutFormat::NHWC;
        // Add restride nodes to all inputs
        auto node_insert = value_input->uses().at(0).user;
        WithInsertPoint insert_point(node_insert);
        auto op_restride = c10::Symbol::fromQualString("hpu::restride_cl");
        auto dims = is_5d_layout
            ? getDimsForLayout5d(
                  habana::LayoutFormat::NHWC, habana::LayoutFormat::NCHW)
            : getDimsForLayout(
                  habana::LayoutFormat::NHWC, habana::LayoutFormat::NCHW);
        auto value_dims = graph->insertConstant(IValue(dims));
        auto restride_node =
            graph->create(op_restride, {value_input, value_dims}, 1);
        restride_node->s_(c10::attr::name, "permute_graph_pass/restide_node");
        restride_node->output(0)->setDebugName(
            value_input->debugName() + "_restrided");
        graph->insertNode(restride_node);
        value_input->replaceAllUsesAfterNodeWith(
            restride_node, restride_node->output(0));
        // mark output layouts
        value_to_tensor_layout[restride_node->output(0)].layout =
            habana::LayoutFormat::NHWC;
        value_to_tensor_layout[restride_node->output(0)].layout_at_graph_entry =
            habana::LayoutFormat::NHWC;
        auto value_in_tt = value_input->type()->cast<TensorType>();
        restride_node->output(0)->setType(c10::TensorType::create(
            value_in_tt->scalarType(),
            value_in_tt->device(),
            value_in_tt->dim(),
            false));
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
    if (win->type()->kind() == c10::TypeKind::TensorType) {
      if (*win->type()->cast<TensorType>()->dim() == 4 ||
          *win->type()->cast<TensorType>()->dim() == 5) {
        value_to_tensor_layout[win].layout = habana::LayoutFormat::HWCK;
        value_to_tensor_layout[win].layout_at_graph_entry =
            habana::LayoutFormat::HWCK;
      }
    }
  }

  std::ostringstream o;
  auto str = o.str();
  for (auto weight : weight_values) {
    std::ostringstream o;
    o << "weight vlaueID: ";
    o << weight->debugName();
    str.append(o.str());
    str.append("\n");
  }
  PT_LAZY_DEBUG(str);

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
          // if already marked by weightmarkingPass, assign the Layout
          if (value_to_tensor_layout.find(value_out) ==
              value_to_tensor_layout.end()) {
            value_to_tensor_layout[value_out].layout =
                habana::LayoutFormat::NCHW;
            value_to_tensor_layout[value_out].layout_at_graph_entry =
                habana::LayoutFormat::NCHW;
          }
        }
      }
      continue;
    }

    // Get kernel MetaData
    auto& device = synapse_helpers::HPURegistrar::get_device();
    synDeviceId device_id = device.id();
    habana::HabanaOperatorPtr habana_kernel = habana::KernelRegistry().get(
        device_id, node->schema().operator_name(), c10::ScalarType::Float);
    TORCH_CHECK(
        habana_kernel, node->schema().operator_name(), " is not registered!");
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

        // View and Index as per original PT layout
        if ((strcmp(node->kind().toQualString(), "aten::view") == 0) ||
            (strcmp(node->kind().toQualString(), "hpu::view") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::expand") == 0) ||
            (strcmp(node->kind().toQualString(), "hpu::expand") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::index") == 0) ||
            (strcmp(node->kind().toQualString(), "hpu::index") == 0)) {
          if ((tensor_layout != habana::LayoutFormat::NCHW) &&
              tensor_layout != habana::LayoutFormat::HWCK) {
            permute_required = true;
            perm_layout = habana::LayoutFormat::NCHW;
          }
        }

        // aten::slice used for static shapes and hpu:slice used for dynamic
        // shapes. aten::slice is optimized for permute pass by having special
        // check for dimBasedOps to reduce number of permutes. hpu::slice cannot
        // be optimized because contiguous shape vectors are prepared at the
        // front end by assuming that the inputs are always contiguous.

        // dim based Ops as per original PT layout NCHW
        if ((strcmp(node->kind().toQualString(), "aten::mean") == 0) ||
            (strcmp(node->kind().toQualString(), "hpu::slice") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::permute") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::select") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::transpose") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::argmax") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::split_with_sizes") ==
             0) ||
            (strcmp(node->kind().toQualString(), "aten::_softmax") == 0) ||
            (strcmp(node->kind().toQualString(), "hpu::sum_dim_IntList") ==
             0) ||
            (strcmp(
                 node->kind().toQualString(), "aten::_softmax_backward_data") ==
             0) ||
            (strcmp(node->kind().toQualString(), "hpu::max_dim") == 0) ||
            (strcmp(
                 node->kind().toQualString(),
                 "aten::_log_softmax_backward_data") == 0) ||
            (strcmp(node->kind().toQualString(), "aten::_log_softmax") == 0)) {
          if ((tensor_layout != habana::LayoutFormat::NCHW) &&
              (tensor_layout != habana::LayoutFormat::HWCK)) {
            permute_required = true;
            perm_layout = habana::LayoutFormat::NCHW;
          }
        }

        // add permutes to memcpy if src and dst formats don't match
        if ((strcmp(
                 node->kind().toQualString(), "hpu::habana_d2d_memcpy_other") ==
             0)) {
          auto value_out = node->input(1);
          auto dst_layout = value_to_tensor_layout[value_out].layout;
          if ((tensor_layout != dst_layout) &&
              tensor_layout != habana::LayoutFormat::HWCK) {
            permute_required = true;
            perm_layout = dst_layout;
          }
        }

        // add permtues in the graph
        if (permute_required) {
          if (*value_in->type()->cast<TensorType>()->dim() == 4) {
            auto dims = getDimsForLayout(perm_layout, tensor_layout);
            anchor_nodes_[node].push_back(std::make_pair(value_in, dims));
            tensor_layout = perm_layout;
          }

          if (*value_in->type()->cast<TensorType>()->dim() == 5) {
            auto dims = getDimsForLayout5d(perm_layout, tensor_layout);
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
        // if already marked by weightmarkingPass, assign the Layout
        if (value_to_tensor_layout.find(value_out) !=
            value_to_tensor_layout.end()) {
          out_layout = value_to_tensor_layout[value_out].layout;
        }
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
      } else if ((strcmp(node->kind().toQualString(), "hpu::cast") == 0)) {
        // special case format info passing cast node
        auto value_in = node->input(0);
        auto in_layout = value_to_tensor_layout[value_in].layout;
        auto in_layout_entry =
            value_to_tensor_layout[value_in].layout_at_graph_entry;
        for (auto value_out : node->outputs()) {
          value_to_tensor_layout[value_out].layout = in_layout;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              in_layout_entry;
        }
      } else if ((strcmp(
                      node->kind().toQualString(),
                      "hpu::habana_d2d_memcpy_other") == 0)) {
        // special case format info passing mem cpy node
        auto value_in = node->input(1);
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
          (strcmp(node->kind().toQualString(), "hpu::slice") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::permute") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::select") == 0) ||
          (strcmp(node->kind().toQualString(), "hpu::view") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::transpose") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::expand") == 0) ||
          (strcmp(node->kind().toQualString(), "hpu::expand") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::argmax") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::split_with_sizes") ==
           0) ||
          (strcmp(node->kind().toQualString(), "hpu::index") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::index") == 0) ||
          (strcmp(node->kind().toQualString(), "hpu::sum_dim_IntList") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::mean") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::_softmax") == 0) ||
          (strcmp(
               node->kind().toQualString(), "aten::_softmax_backward_data") ==
           0) ||
          (strcmp(node->kind().toQualString(), "hpu::max_dim") == 0) ||
          (strcmp(
               node->kind().toQualString(),
               "aten::_log_softmax_backward_data") == 0) ||
          (strcmp(node->kind().toQualString(), "aten::_log_softmax") == 0)) {
        // View() layout is always NCHW as per original PT format
        // [ToDo] consider case permute_cl followed by view()
        // %1 = aten::permute_cl(...)
        // %2 = aten::view(%1)
        auto value_in = node->input(0);
        for (auto value_out : node->outputs()) {
          if (!weight_pass.isMarkedAsweight(value_out)) {
            value_to_tensor_layout[value_out].layout =
                habana::LayoutFormat::NCHW;
            value_to_tensor_layout[value_out].layout_at_graph_entry =
                value_to_tensor_layout[value_in].layout_at_graph_entry;
          }
        }
      } else if (isDimBasedOp(node)) {
        auto value_in = node->input(0);
        auto dimIdx = dimBasedOpsIdx.at(node->kind().toQualString());
        auto dim = toIValue(node->input(dimIdx))->toInt();
        auto value_layout_entry =
            value_to_tensor_layout[value_in].layout_at_graph_entry;
        auto tensor_layout = value_to_tensor_layout[value_in].layout;
        for (auto value_out : node->outputs()) {
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              value_layout_entry;
          value_to_tensor_layout[value_out].layout = tensor_layout;
          auto is_5d_layout =
              *value_out->type()->cast<TensorType>()->dim() == 5;
          auto layout_dim = is_5d_layout ? getLayoutDim5d(tensor_layout, dim)
                                         : getLayoutDim(tensor_layout, dim);
          if ((tensor_layout != habana::LayoutFormat::NCHW) &&
              (tensor_layout != habana::LayoutFormat::HWCK)) {
            if (*value_out->type()->cast<TensorType>()->dim() == 4 ||
                is_5d_layout) {
              // if dims can not be adjusted bring back to PT layout
              if (layout_dim != dim) {
                WithInsertPoint insert_point(node);
                auto value_dim = graph->insertConstant(IValue(layout_dim));
                node->replaceInputWith(node->input(dimIdx), value_dim);
              } else {
                auto value_in = node->input(0);
                auto dims = is_5d_layout
                    ? getDimsForLayout5d(
                          habana::LayoutFormat::NCHW,
                          value_to_tensor_layout[value_in].layout)
                    : getDimsForLayout(
                          habana::LayoutFormat::NCHW,
                          value_to_tensor_layout[value_in].layout);
                anchor_nodes_[node].push_back(std::make_pair(value_in, dims));
                value_to_tensor_layout[value_out].layout =
                    habana::LayoutFormat::NCHW;
              }
            } else {
              value_to_tensor_layout[value_out].layout =
                  habana::LayoutFormat::NCHW;
            }
          }
        }
      } else if ((strcmp(node->kind().toQualString(), "aten::cat") == 0)) {
        auto tListNode = node->input(0)->node();
        auto value_in0 = tListNode->input(0);
        auto value_out = node->output(0);
        auto dimIdx = 1; // position of dim input
        auto dim = toIValue(node->input(dimIdx))->toInt();
        // check if all inputs are 4d and are in NHWC format
        auto allNHWC = true;
        auto is_5d_layout = false;
        for (auto value_in : tListNode->inputs()) {
          is_5d_layout = *value_in->type()->cast<TensorType>()->dim() == 5;
          if ((value_to_tensor_layout[value_in].layout ==
               habana::LayoutFormat::NCHW) ||
              (*value_in->type()->cast<TensorType>()->dim() < 4)) {
            allNHWC = false;
            break;
          }
        }
        // if all inputs are 4d and NHWC then adding permutes at inputs not
        // needed, instead change dim
        if (allNHWC) {
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              value_to_tensor_layout[value_in0].layout_at_graph_entry;
          auto layout_dim = is_5d_layout
              ? getLayoutDim5d(value_to_tensor_layout[value_in0].layout, dim)
              : getLayoutDim(value_to_tensor_layout[value_in0].layout, dim);
          WithInsertPoint insert_point(node);
          auto value_dim = graph->insertConstant(IValue(layout_dim));
          node->replaceInputWith(node->input(dimIdx), value_dim);
        }
        // otherwise go through inputs and insert permutes to go to NCHW
        else {
          value_to_tensor_layout[value_out].layout = habana::LayoutFormat::NCHW;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              value_to_tensor_layout[value_in0].layout_at_graph_entry;
          for (auto value_in : tListNode->inputs()) {
            if (value_to_tensor_layout[value_in].layout !=
                habana::LayoutFormat::NCHW) {
              if (*value_in->type()->cast<TensorType>()->dim() == 4) {
                auto dims = getDimsForLayout(
                    habana::LayoutFormat::NCHW,
                    value_to_tensor_layout[value_in].layout);
                anchor_nodes_[tListNode].push_back(
                    std::make_pair(value_in, dims));
              }

              if (is_5d_layout) {
                auto dims = getDimsForLayout5d(
                    habana::LayoutFormat::NCHW,
                    value_to_tensor_layout[value_in].layout);
                anchor_nodes_[tListNode].push_back(
                    std::make_pair(value_in, dims));
              }
            }
          }
        }
      } else {
        if (strcmp(node->kind().toQualString(), "aten::constant_pad_nd") == 0) {
          auto value_in0 = node->input(0);
          // PRefix the pad value with 0,0 tuple so that padding of 'C' is
          // skipped and padding is applied to W, H
          if (value_to_tensor_layout[value_in0].layout ==
              habana::LayoutFormat::NHWC) {
            auto const padIdx = 1;
            auto pad = toIValue(node->input(padIdx))->toIntList().vec();
            std::vector<int64_t> pad_including_C(pad.size() + 2);
            pad_including_C[0] = pad_including_C[1] = 0;
            for (unsigned int i = 0; i < pad.size(); i++) {
              pad_including_C[i + 2] = pad[i];
            }
            WithInsertPoint insert_point(node);
            auto value_dim =
                graph->insertConstant(IValue(at::IntArrayRef(pad_including_C)));
            node->replaceInputWith(node->input(padIdx), value_dim);
          }
        }

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
          if (value_to_tensor_layout.find(value_out) !=
              value_to_tensor_layout.end()) {
            assigned_input_layout = value_to_tensor_layout[value_out].layout;
            origin_input_layout =
                value_to_tensor_layout[value_out].layout_at_graph_entry;
          }
          value_to_tensor_layout[value_out].layout = assigned_input_layout;
          value_to_tensor_layout[value_out].layout_at_graph_entry =
              origin_input_layout;
          if (assigned_input_layout == habana::LayoutFormat::HWCK) {
            if (is_4d_5d_value(value_out)) {
              value_to_tensor_layout[value_out].layout = assigned_input_layout;
              value_to_tensor_layout[value_out].layout_at_graph_entry =
                  assigned_input_layout;
            } else {
              value_to_tensor_layout[value_out].layout =
                  habana::LayoutFormat::NCHW;
              value_to_tensor_layout[value_out].layout_at_graph_entry =
                  habana::LayoutFormat::NCHW;
            }
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
    auto is_5d_layout = *value_out->type()->cast<TensorType>()->dim() == 5;
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
          } else if (is_5d_layout) {
            at::IntArrayRef dims;
            static const int64_t dimarr[] = {0, 4, 1, 2, 3};
            dims = dimarr;
            anchor_restride_nodes_[node_return].push_back(
                std::make_pair(value_out, dims));
          } else {
            at::IntArrayRef dims;
            static const int64_t dimarr[] = {0, 3, 1, 2};
            dims = dimarr;
            if (*value_out->type()->cast<TensorType>()->dim() == 4) {
              anchor_restride_nodes_[node_return].push_back(
                  std::make_pair(value_out, dims));
            }
          }
        }
      } else {
        if (prev_layout != cur_layout) {
          if (*value_out->type()->cast<TensorType>()->dim() == 4) {
            auto dims = getDimsForLayout(prev_layout, cur_layout);
            anchor_nodes_[node_return].push_back(
                std::make_pair(value_out, dims));
          }

          if (is_5d_layout) {
            auto dims = getDimsForLayout5d(prev_layout, cur_layout);
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
