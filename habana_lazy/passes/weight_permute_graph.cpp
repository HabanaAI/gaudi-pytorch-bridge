/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "weight_permute_graph.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "pass_utils.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"
#include "torch/csrc/jit/ir/ir.h"
using namespace torch::jit;
namespace habana_lazy {

bool isInGraphInputs(
    std::shared_ptr<Graph>& graph,
    const torch::jit::Value* value) {
  auto graph_ins = graph->inputs();
  for (auto value_in : graph_ins) {
    if (value->unique() == value_in->unique()) {
      return true;
    }
  }
  return false;
}

size_t getValuePosInStack(
    std::shared_ptr<Graph>& graph,
    const torch::jit::Value* value) {
  auto graph_ins = graph->inputs();
  size_t idx = 0;
  for (auto value_in : graph_ins) {
    if (value->unique() == value_in->unique()) {
      return idx;
    }
    idx++;
  }
  return -1;
}

bool isD2DCopyOther(const torch::jit::Value* value) {
  for (auto u : value->uses()) {
    auto node = u.user;
    if (strcmp(node->kind().toQualString(), "hpu::habana_d2d_memcpy_other") ==
        0) {
      return true;
    }
  }
  return false;
}

bool isControlEdge(const torch::jit::Value* value) {
  for (auto u : value->uses()) {
    auto node = u.user;
    if (strcmp(node->kind().toQualString(), "hpu::control_edge_") == 0) {
      return true;
    }
  }
  return false;
}

bool isAsStridedNode(const torch::jit::Value* value_in) {
  if (strcmp(
          value_in->node()->kind().toQualString(), "hpu::as_strided_lazy_") ==
      0) {
    return true;
  }
  return false;
}

bool is_4d_5d_tensor(const at::Tensor tensor) {
  return (tensor.dim() == 4 || tensor.dim() == 5) ? true : false;
}

at::IntArrayRef getDimsForWeightLayout(
    habana::LayoutFormat channel_order,
    habana::LayoutFormat current_order,
    unsigned tensor_dims) {
  at::IntArrayRef dims;

  // using NCHW/NHWC/HWCK since synapse 5d layout nomenclature
  // is not clear. Note that for 5d layout channel dim is 1 for
  // NCDHW and 4 for NDHWC
  if (current_order == habana::LayoutFormat::NCHW) {
    if (channel_order == habana::LayoutFormat::HWCK) {
      if (tensor_dims == 5) {
        static const int64_t dimarr[] = {2, 3, 4, 1, 0};
        dims = dimarr;
      } else if (tensor_dims == 4) {
        static const int64_t dimarr[] = {2, 3, 1, 0};
        dims = dimarr;
      } else {
        TORCH_CHECK(
            0,
            " InsertWeightPermute_graph: Weight permute called for unsupported channel order");
      }
    } else {
      TORCH_CHECK(
          0,
          " InsertWeightPermute_graph: Weight permute called for unsupported channel order");
    }
  } else if (current_order == habana::LayoutFormat::HWCK) {
    if (channel_order == habana::LayoutFormat::NCHW) {
      if (tensor_dims == 5) {
        static const int64_t dimarr[] = {4, 3, 0, 1, 2};
        dims = dimarr;
      } else if (tensor_dims == 4) {
        static const int64_t dimarr[] = {3, 2, 0, 1};
        dims = dimarr;
      } else {
        TORCH_CHECK(
            0,
            " InsertWeightPermute_graph: Weight permute called for unsupported channel order");
      }
    } else {
      TORCH_CHECK(
          0,
          " InsertWeightPermute_graph: Weight permute called for unsupported channel order");
    }
  } else {
    TORCH_CHECK(
        0,
        " InsertWeightPermute_graph: Weight permute called for unsupported channel order");
  }

  return dims;
}

void WeightPermutesEagerMode(std::shared_ptr<Graph>& graph) {
  std::vector<torch::jit::Node*> remove_nodes;
  for (auto node : graph->nodes()) {
    if (strcmp(node->kind().toQualString(), "aten::convolution_overrideable") ==
        0) {
      auto value_in = node->input(1);
      WithInsertPoint insert_point(node);
      auto op_permute = c10::Symbol::fromQualString("aten::permute");
      torch::jit::Value* value_dims1 = nullptr;
      if (*value_in->type()->cast<TensorType>()->dim() == 4) {
        static const int64_t dimarr[] = {2, 3, 1, 0};
        at::IntArrayRef dims1 = dimarr;
        value_dims1 = graph->insertConstant(IValue(dims1));
      } else if (*value_in->type()->cast<TensorType>()->dim() == 5) {
        static const int64_t dimarr[] = {2, 3, 4, 1, 0};
        at::IntArrayRef dims1 = dimarr;
        value_dims1 = graph->insertConstant(IValue(dims1));
      } else {
        HABANA_ASSERT(0);
      }
      auto permute_node = graph->create(op_permute, {value_in, value_dims1}, 1);
      graph->insertNode(permute_node);
      node->replaceInputWith(value_in, permute_node->output(0));
    }
    if (strcmp(
            node->kind().toQualString(),
            "aten::convolution_backward_overrideable") == 0) {
      // input
      auto value_in = node->input(2);
      WithInsertPoint insert_point(node);
      auto op_permute = c10::Symbol::fromQualString("aten::permute");
      torch::jit::Value* value_dims1 = nullptr;
      if (*value_in->type()->cast<TensorType>()->dim() == 4) {
        static const int64_t dimarr[] = {2, 3, 1, 0};
        at::IntArrayRef dims1 = dimarr;
        value_dims1 = graph->insertConstant(IValue(dims1));
      } else if (*value_in->type()->cast<TensorType>()->dim() == 5) {
        static const int64_t dimarr[] = {2, 3, 4, 1, 0};
        at::IntArrayRef dims1 = dimarr;
        value_dims1 = graph->insertConstant(IValue(dims1));
      } else {
        HABANA_ASSERT(0);
      }
      auto permute_node = graph->create(op_permute, {value_in, value_dims1}, 1);
      graph->insertNode(permute_node);
      node->replaceInputWith(value_in, permute_node->output(0));

      auto value_out = node->output(1);
      auto op_permute2 = c10::Symbol::fromQualString("aten::permute");
      torch::jit::Value* value_dims2 = nullptr;
      if (*value_out->type()->cast<TensorType>()->dim() == 4) {
        static const int64_t dimarr[] = {3, 2, 0, 1};
        at::IntArrayRef dims2 = dimarr;
        value_dims2 = graph->insertConstant(IValue(dims2));
      } else if (*value_out->type()->cast<TensorType>()->dim() == 5) {
        static const int64_t dimarr[] = {4, 3, 0, 1, 2};
        at::IntArrayRef dims2 = dimarr;
        value_dims2 = graph->insertConstant(IValue(dims2));
      } else {
        HABANA_ASSERT(0);
      }
      auto permute_node2 =
          graph->create(op_permute2, {value_out, value_dims2}, 1);
      permute_node2->insertAfter(node);
      value_out->replaceAllUsesAfterNodeWith(
          permute_node2, permute_node2->output(0));
    }
  }
}

void InsertWeightRestride_graph(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack) {
  // Remove permute Nodes to replace with inplace permutes
  std::vector<torch::jit::Node*> remove_nodes;
  for (auto node : graph->nodes()) {
    if ((strcmp(node->kind().toQualString(), "hpu::permute_weight") == 0) ||
        (strcmp(node->kind().toQualString(), "hpu::permuted_weight_restride") ==
         0)) {
      remove_nodes.push_back(node);
    }
  }
  for (auto node : remove_nodes) {
    auto value_in = node->input(0);
    auto value_out = node->output(0);
    value_out->replaceAllUsesAfterNodeWith(node, value_in);
    node->removeInput(1);
    node->destroy();
  }

  // mark already permuted weights
  WeightIdentificationPass weight_pass;
  auto graph_inputs = graph->inputs();
  for (auto value_in : graph_inputs) {
    auto value_idx = getValuePosInStack(graph, value_in);
    HABANA_ASSERT(value_idx + 1);
    if (stack[value_idx].isTensor()) {
      auto tensor = stack[value_idx].toTensor();
      if (tensor.has_storage() && is_4d_5d_tensor(tensor)) {
        auto hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
        auto layout_format = hb_tensor->GetTensorLayout();
        if (layout_format == habana_lazy::LayoutFormat::kHWCK) {
          weight_pass.weightMarker(value_in);
        }
      }
    }
  }

  {
    auto weight_values = weight_pass.getWeightTensors();
    std::ostringstream o;
    auto str = o.str();
    for (auto weight : weight_values) {
      std::ostringstream o;
      o << "Already Marked weight vlaueID: ";
      o << weight->debugName();
      str.append(o.str());
      str.append("\n");
    }
    PT_LAZY_DEBUG(str);
  }

  // mark customKernel Weights
  auto OptimKernls = weight_pass.getCustomOptimizerWeights();
  for (auto node : graph->nodes()) {
    auto node_str = node->kind().toQualString();
    if (OptimKernls.count(node_str)) {
      auto allIdx = OptimKernls[node_str];
      for (auto idx : allIdx) {
        auto in_val = node->input(idx);
        if (strcmp(
                in_val->node()->kind().toQualString(), "prim::ListConstruct") ==
            0) {
          for (auto list_input_val : in_val->node()->inputs()) {
            torch::jit::Value* value_in = nullptr;
            if (isInGraphInputs(graph, list_input_val)) {
              value_in = list_input_val;
            } else {
              auto list_input_node = list_input_val->node();
              if (isInGraphInputs(graph, list_input_node->input(0))) {
                value_in = list_input_node->input(0);
              }
            }
            if (value_in) {
              auto value_idx = getValuePosInStack(graph, value_in);
              HABANA_ASSERT(value_idx + 1);
              if (stack[value_idx].isTensor()) {
                auto tensor = stack[value_idx].toTensor();
                if (is_4d_5d_tensor(tensor)) {
                  weight_pass.weightMarker(value_in);
                }
              }
            }
          }
        }
      }
      for (auto output : node->outputs()) {
        auto uses = output->uses();
        for (auto u : uses) {
          auto output_node = u.user;
          if (strcmp(output_node->kind().toQualString(), "prim::ListUnpack") ==
              0) {
            for (auto list_output_val : output_node->outputs()) {
              if (*list_output_val->type()->cast<TensorType>()->dim() == 4)
                weight_pass.weightMarker(list_output_val);
            }
          }
        }
      }
    }
  }

  // mark weights of full graph
  weight_pass.markWeightTensors(graph, true);
  auto weight_values = weight_pass.getWeightTensors();

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

  std::map<torch::jit::Value*, torch::jit::Node*> d2d_memcpy_nodes_;
  // In-place weight permute loop
  for (auto weight_value : weight_values) {
    if (isInGraphInputs(graph, weight_value)) {
      auto value_idx = getValuePosInStack(graph, weight_value);
      HABANA_ASSERT(value_idx + 1);
      if (stack[value_idx].isTensor()) {
        auto tensor = stack[value_idx].toTensor();
        if (!is_4d_5d_tensor(tensor))
          continue;
        auto hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
        auto layout_format = hb_tensor->GetTensorLayout();
        if ((isD2DCopyOther(weight_value) || (isControlEdge(weight_value)))) {
          hb_tensor->SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
          layout_format = habana_lazy::LayoutFormat::kHWCK;
        }
        if (layout_format != habana_lazy::LayoutFormat::kHWCK) {
          auto value_in = const_cast<torch::jit::Value*>(weight_value);

          auto op_permute = c10::Symbol::fromQualString("aten::permute");
          auto dims1 = getDimsForWeightLayout(
              habana::LayoutFormat::HWCK,
              habana::LayoutFormat::NCHW,
              tensor.dim());
          auto value_dims1 = graph->insertConstant(IValue(dims1));
          auto permute_node =
              graph->create(op_permute, {value_in, value_dims1}, 1);
          permute_node->insertAfter(value_in->node());
          value_dims1->node()->moveBefore(permute_node);

          auto op_control_edge1 =
              c10::Symbol::fromQualString("hpu::control_edge_");
          auto control_edge_node1 =
              graph->create(op_control_edge1, {value_in}, 1);
          control_edge_node1->insertAfter(value_in->node());

          auto op_as_strided1 =
              c10::Symbol::fromQualString("hpu::as_strided_layout_");
          auto dims2 = getDimsForWeightLayout(
              habana::LayoutFormat::HWCK,
              habana::LayoutFormat::NCHW,
              tensor.dim());
          auto value_dims2 = graph->insertConstant(IValue(dims2));
          auto as_strided_node1 = graph->create(
              op_as_strided1, {control_edge_node1->output(0), value_dims2}, 1);
          as_strided_node1->insertAfter(control_edge_node1);
          value_dims2->node()->moveBefore(as_strided_node1);

          auto op_control_edge2 =
              c10::Symbol::fromQualString("hpu::control_edge_");
          auto control_edge_node2 =
              graph->create(op_control_edge2, {as_strided_node1->output(0)}, 1);
          control_edge_node2->insertAfter(as_strided_node1);

          auto op_d2d_copy =
              c10::Symbol::fromQualString("hpu::habana_d2d_memcpy_other");
          auto d2d_copy_node = graph->create(
              op_d2d_copy,
              {permute_node->output(0), control_edge_node2->output(0)},
              1);
          d2d_copy_node->insertAfter(permute_node);

          value_in->replaceAllUsesAfterNodeWith(
              d2d_copy_node, d2d_copy_node->output(0));
          hb_tensor->SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
        } else {
          // add restride NCHW -> HWCK
          auto value_in = const_cast<torch::jit::Value*>(weight_value);
          auto op_restride = c10::Symbol::fromQualString("hpu::restride");
          auto dims = getDimsForWeightLayout(
              habana::LayoutFormat::HWCK,
              habana::LayoutFormat::NCHW,
              tensor.dim());
          auto value_dims = graph->insertConstant(IValue(dims));
          auto restride_node =
              graph->create(op_restride, {value_in, value_dims}, 1);
          restride_node->insertAfter(value_in->node());
          value_dims->node()->moveBefore(restride_node);
          {
            auto value_in_tt = value_in->type()->cast<TensorType>();
            restride_node->output(0)->setType(c10::TensorType::create(
                value_in_tt->scalarType(),
                value_in_tt->device(),
                value_in_tt->dim(),
                false));
          }
          value_in->replaceAllUsesAfterNodeWith(
              restride_node, restride_node->output(0));
        }
      }
    } else {
      auto value_in = const_cast<torch::jit::Value*>(weight_value);
      if (value_in->type()->kind() == c10::TypeKind::TensorType) {
        if (is_4d_5d_value(value_in)) {
          if (isAsStridedNode(value_in)) {
            auto node = value_in->node();
            auto sizes = toIValue(node->input(1))->toIntVector();
            auto is_5d_layout = sizes.size() == 5 ? true : false;
            int64_t dim_out_pos[] = {2, 3, 1, 0};
            int64_t dim_out_pos_3d[] = {2, 3, 4, 1, 0};
            at::IntArrayRef out_pos;
            out_pos = dim_out_pos;
            if (is_5d_layout)
              out_pos = dim_out_pos_3d;

            auto new_pos = out_pos.vec();
            std::vector<long int> swapped_sizes = {
                sizes[new_pos[0]],
                sizes[new_pos[1]],
                sizes[new_pos[2]],
                sizes[new_pos[3]]};
            if (is_5d_layout) {
              swapped_sizes.push_back(sizes[new_pos[4]]);
            }

            std::vector<long int> strides = {
                swapped_sizes[1] * swapped_sizes[2] * swapped_sizes[3],
                swapped_sizes[3] * swapped_sizes[2],
                swapped_sizes[3],
                1};
            if (is_5d_layout) {
              strides.clear();
              strides.push_back(
                  swapped_sizes[4] * swapped_sizes[3] * swapped_sizes[2] *
                  swapped_sizes[1]);
              strides.push_back(
                  swapped_sizes[4] * swapped_sizes[3] * swapped_sizes[2]);
              strides.push_back(swapped_sizes[4] * swapped_sizes[3]);
              strides.push_back(swapped_sizes[4]);
              strides.push_back(1);
            }

            WithInsertPoint insert_point(node);
            auto value_dim = graph->insertConstant(IValue(swapped_sizes));
            auto value_strides = graph->insertConstant(IValue(strides));
            node->replaceInputWith(node->input(1), value_dim);
            node->replaceInputWith(node->input(2), value_strides);
          }
        }
      }
    }
  }

  auto node_return = graph->return_node();
  for (auto weight_value : weight_values) {
    if (isRetunrOut(graph, weight_value)) {
      if (!is_4d_5d_value(weight_value))
        continue;
      // add restride HWCk -> NCHW
      auto value_in = const_cast<torch::jit::Value*>(weight_value);
      WithInsertPoint insert_point(node_return);
      auto op_restride = c10::Symbol::fromQualString("hpu::restride");
      auto dims = getDimsForWeightLayout(
          habana::LayoutFormat::NCHW,
          habana::LayoutFormat::HWCK,
          *weight_value->type()->cast<TensorType>()->dim());
      auto value_dims = graph->insertConstant(IValue(dims));
      auto restride_node =
          graph->create(op_restride, {value_in, value_dims}, 1);
      graph->insertNode(restride_node);
      value_dims->node()->moveBefore(restride_node);
      {
        auto value_in_tt = value_in->type()->cast<TensorType>();
        restride_node->output(0)->setType(c10::TensorType::create(
            value_in_tt->scalarType(),
            value_in_tt->device(),
            value_in_tt->dim(),
            false));
      }
      value_in->replaceAllUsesAfterNodeWith(
          restride_node, restride_node->output(0));

      auto d2d_value_out = const_cast<torch::jit::Value*>(weight_value);
      if (d2d_memcpy_nodes_.find(d2d_value_out) != d2d_memcpy_nodes_.end()) {
        auto node = d2d_memcpy_nodes_[d2d_value_out];
        WithInsertPoint insert_point(node);

        auto op_fill = c10::Symbol::fromQualString("aten::fill_");
        static const int64_t fill_zero = 0;
        auto value_zero = graph->insertConstant(IValue(fill_zero));
        auto fill_node =
            graph->create(op_fill, {node->input(1), value_zero}, 1);
        graph->insertNode(fill_node);

        auto op_add_ = c10::Symbol::fromQualString("aten::add_");
        static const int64_t constant = 1;
        auto value_const = graph->insertConstant(IValue(constant));
        auto add_node_ = graph->create(
            op_add_, {fill_node->output(0), node->input(0), value_const}, 1);
        graph->insertNode(add_node_);
        d2d_value_out->replaceAllUsesAfterNodeWith(
            add_node_, add_node_->output(0));
        node->destroy();
      }
    }
  }
}

void InsertWeightPermute_graph(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack) {
  // eager weight permutes
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 1) {
    WeightPermutesEagerMode(graph);
    return;
  }

  // weight restrieds only
  InsertWeightRestride_graph(graph, stack);
  return;

  // Remove permute Nodes to replace with inplace permutes
  std::vector<torch::jit::Node*> remove_nodes;
  for (auto node : graph->nodes()) {
    if ((strcmp(node->kind().toQualString(), "hpu::permute_weight") == 0) ||
        (strcmp(node->kind().toQualString(), "hpu::permuted_weight_restride") ==
         0)) {
      remove_nodes.push_back(node);
    }
  }
  for (auto node : remove_nodes) {
    auto value_in = node->input(0);
    auto value_out = node->output(0);
    value_out->replaceAllUsesAfterNodeWith(node, value_in);
    node->destroy();
  }

  // mark already permuted weights
  WeightIdentificationPass weight_pass;
  auto graph_inputs = graph->inputs();
  for (auto value_in : graph_inputs) {
    auto value_idx = getValuePosInStack(graph, value_in);
    HABANA_ASSERT(value_idx + 1);
    if (stack[value_idx].isTensor()) {
      auto tensor = stack[value_idx].toTensor();
      if (tensor.has_storage()) {
        auto hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
        auto layout_format = hb_tensor->GetTensorLayout();
        if (layout_format == habana_lazy::LayoutFormat::kHWCK) {
          weight_pass.weightMarker(value_in);
        }
      }
    }
  }

  // mark customKernel Weights
  auto OptimKernls = weight_pass.getCustomOptimizerWeights();
  for (auto node : graph->nodes()) {
    auto node_str = node->kind().toQualString();
    if (OptimKernls.count(node_str)) {
      auto allIdx = OptimKernls[node_str];
      for (auto idx : allIdx) {
        auto in_val = node->input(idx);
        if (strcmp(
                in_val->node()->kind().toQualString(), "prim::ListConstruct") ==
            0) {
          for (auto list_input_val : in_val->node()->inputs()) {
            torch::jit::Value* value_in = nullptr;
            if (isInGraphInputs(graph, list_input_val)) {
              value_in = list_input_val;
            } else {
              auto list_input_node = list_input_val->node();
              if (isInGraphInputs(graph, list_input_node->input(0))) {
                value_in = list_input_node->input(0);
              }
            }
            if (value_in) {
              auto value_idx = getValuePosInStack(graph, value_in);
              HABANA_ASSERT(value_idx + 1);
              if (stack[value_idx].isTensor()) {
                auto tensor = stack[value_idx].toTensor();
                if (tensor.dim() == 4) {
                  weight_pass.weightMarker(value_in);
                }
              }
            }
          }
        }
      }
    }
  }

  // mark weights of full graph
  weight_pass.markWeightTensors(graph, true);
  auto weight_values = weight_pass.getWeightTensors();

  // In-place weight permute loop
  for (auto weight_value : weight_values) {
    if (isInGraphInputs(graph, weight_value)) {
      auto value_idx = getValuePosInStack(graph, weight_value);
      HABANA_ASSERT(value_idx + 1);
      if (stack[value_idx].isTensor()) {
        auto tensor = stack[value_idx].toTensor();
        auto hb_tensor = habana_lazy::GetHbInternalTensorImpl(tensor);
        auto layout_format = hb_tensor->GetTensorLayout();
        if (layout_format != habana_lazy::LayoutFormat::kHWCK) {
          auto value_in = const_cast<torch::jit::Value*>(weight_value);
          if (isD2DCopyOther(weight_value)) {
            hb_tensor->SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
            continue;
          }

          auto op_permute = c10::Symbol::fromQualString("aten::permute");
          at::IntArrayRef dims1 = {2, 3, 1, 0};
          auto value_dims1 = graph->insertConstant(IValue(dims1));
          auto permute_node =
              graph->create(op_permute, {value_in, value_dims1}, 1);
          permute_node->s_(c10::attr::name, "permute_node");
          permute_node->outputs()[0]->setDebugName(
              value_in->debugName() + "/premute");
          permute_node->insertAfter(value_in->node());

          auto op_control_edge1 =
              c10::Symbol::fromQualString("hpu::control_edge_");
          auto control_edge_node1 =
              graph->create(op_control_edge1, {value_in}, 1);
          control_edge_node1->s_(c10::attr::name, "control_edge_node1");
          control_edge_node1->insertAfter(value_in->node());

          auto op_as_strided1 =
              c10::Symbol::fromQualString("hpu::as_strided_layout_");
          at::IntArrayRef dims2 = {2, 3, 1, 0};
          auto value_dims2 = graph->insertConstant(IValue(dims2));
          auto as_strided_node1 = graph->create(
              op_as_strided1, {control_edge_node1->output(0), value_dims2}, 1);
          as_strided_node1->s_(c10::attr::name, "as_strided_node1");
          as_strided_node1->insertAfter(control_edge_node1);

          auto op_control_edge2 =
              c10::Symbol::fromQualString("hpu::control_edge_");
          auto control_edge_node2 =
              graph->create(op_control_edge2, {as_strided_node1->output(0)}, 1);
          control_edge_node2->s_(c10::attr::name, "control_edge_node2");
          control_edge_node2->insertAfter(as_strided_node1);

          auto op_d2d_copy =
              c10::Symbol::fromQualString("hpu::habana_d2d_memcpy_other");
          auto d2d_copy_node = graph->create(
              op_d2d_copy,
              {permute_node->output(0), control_edge_node2->output(0)},
              1);
          d2d_copy_node->s_(c10::attr::name, "d2d_copy_node");
          d2d_copy_node->insertAfter(permute_node);

          auto op_control_edge3 =
              c10::Symbol::fromQualString("hpu::control_edge_");
          auto control_edge_node3 =
              graph->create(op_control_edge3, {d2d_copy_node->output(0)}, 1);
          control_edge_node3->s_(c10::attr::name, "control_edge_node3");
          control_edge_node3->insertAfter(d2d_copy_node);

          auto op_as_strided2 =
              c10::Symbol::fromQualString("hpu::as_strided_layout_");
          at::IntArrayRef dims3 = {3, 2, 0, 1};
          auto value_dims3 = graph->insertConstant(IValue(dims3));
          auto as_strided_node2 = graph->create(
              op_as_strided2, {control_edge_node3->output(0), value_dims3}, 1);
          as_strided_node2->s_(c10::attr::name, "as_strided_node2");
          as_strided_node2->insertAfter(control_edge_node3);

          value_in->replaceAllUsesAfterNodeWith(
              as_strided_node2, as_strided_node2->output(0));
          hb_tensor->SetTensorLayout(habana_lazy::LayoutFormat::kHWCK);
        }
      }
    }
  }

  // change i/p and o/p strides of conv nodes only (include hpu::cast nodes)
  torch::jit::graph_node_list graph_nodes = graph->nodes();
  std::vector<Node*> conv_nodes;
  std::map<torch::jit::Value*, std::vector<torch::jit::Node*>>
      conv_weight_nodes_;
  auto kernelInWeightIdx = weight_pass.getConvKernelInWeights();
  for (auto* node : graph_nodes) {
    if ((strcmp(
             node->kind().toQualString(), "aten::convolution_overrideable") ==
         0) ||
        (strcmp(
             node->kind().toQualString(),
             "aten::convolution_backward_overrideable") == 0)) {
      auto idx = kernelInWeightIdx[node->kind().toQualString()];
      auto value_in = node->input(idx);
      if (strcmp(value_in->node()->kind().toQualString(), "hpu::cast") == 0) {
        auto case_node = value_in->node();
        auto cast_node_in = case_node->input(0);
        if (conv_weight_nodes_.find(cast_node_in) == conv_weight_nodes_.end()) {
          std::vector<torch::jit::Node*> cast_nodes;
          cast_nodes.push_back(case_node);
          conv_weight_nodes_[cast_node_in] = cast_nodes;
        }
      } else {
        if (conv_weight_nodes_.find(value_in) == conv_weight_nodes_.end()) {
          std::vector<torch::jit::Node*> conv_nodes;
          auto uses = value_in->uses();
          conv_nodes.push_back(node);
          for (auto u : uses) {
            auto perm_node = u.user;
            if ((strcmp(
                     perm_node->kind().toQualString(),
                     "aten::convolution_overrideable") == 0) ||
                (strcmp(
                     perm_node->kind().toQualString(),
                     "aten::convolution_backward_overrideable") == 0)) {
              if (std::find(conv_nodes.begin(), conv_nodes.end(), perm_node) ==
                  conv_nodes.end())
                conv_nodes.push_back(perm_node);
            }
          }
          if (conv_nodes.size()) {
            conv_weight_nodes_[value_in] = conv_nodes;
          }
        }
      }
    }
  }

  for (auto node_map : conv_weight_nodes_) {
    conv_nodes = node_map.second;
    Node* permute_strided = nullptr;
    for (auto conv_node : conv_nodes) {
      auto idx = kernelInWeightIdx[conv_node->kind().toQualString()];
      if (permute_strided == nullptr) {
        auto value_in = conv_node->input(idx);
        WithInsertPoint insert_point(conv_node);

        auto op_control_edge10 =
            c10::Symbol::fromQualString("hpu::control_edge_");
        auto control_edge_node10 =
            graph->create(op_control_edge10, {value_in}, 1);
        control_edge_node10->s_(c10::attr::name, "control_edge10");
        graph->insertNode(control_edge_node10);

        auto op_as_strided =
            c10::Symbol::fromQualString("hpu::as_strided_layout_");
        static const int64_t dimarr[] = {2, 3, 1, 0};
        at::IntArrayRef dims = dimarr;
        auto value_dims = graph->insertConstant(IValue(dims));
        permute_strided = graph->create(
            op_as_strided, {control_edge_node10->output(0), value_dims}, 1);
        permute_strided->s_(c10::attr::name, "permute_strided");
        graph->insertNode(permute_strided);
        conv_node->replaceInputWith(value_in, permute_strided->output(0));
      } else {
        auto value_in = conv_node->input(idx);
        conv_node->replaceInputWith(value_in, permute_strided->output(0));
      }
    }
  }
  for (auto* conv_node : graph_nodes) {
    if (strcmp(
            conv_node->kind().toQualString(),
            "aten::convolution_backward_overrideable") == 0) {
      auto value_out = conv_node->output(1);
      auto u = value_out->uses();
      auto first_node_use = u.at(0).user;
      if (strcmp(first_node_use->kind().toQualString(), "hpu::cast") == 0) {
        value_out = first_node_use->output(0);
        auto u = value_out->uses();
        first_node_use = u.at(0).user;
      }
      WithInsertPoint insert_point_out(first_node_use);
      auto op_as_strided_out =
          c10::Symbol::fromQualString("hpu::as_strided_layout_");
      static const int64_t dimarr_out[] = {3, 2, 0, 1};
      at::IntArrayRef dims_out = dimarr_out;
      auto value_dims_out = graph->insertConstant(IValue(dims_out));
      auto as_strided_node_out =
          graph->create(op_as_strided_out, {value_out, value_dims_out}, 1);
      as_strided_node_out->s_(c10::attr::name, "as_strided_node_out");
      graph->insertNode(as_strided_node_out);
      value_out->replaceAllUsesAfterNodeWith(
          as_strided_node_out, as_strided_node_out->output(0));
    }
  }
}
} // namespace habana_lazy
