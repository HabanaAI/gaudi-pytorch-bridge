/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "habana_eager/eager_view.h"
#include <cstddef>
#include <cstdint>

namespace habana {
namespace eager {

namespace {

/* In general, inplace ops  read from input tensor and then write to the same
tensor.
The below list of ops ignore the values in the input tensor and overwrite the
contents*/
std::unordered_set<std::string> underscored_ops_reported_as_non_inplace = {
    "aten::zero_",
    "aten::fill_",
    "hpu::bernoulli_",
    "hpu::uniform_",
    "hpu::random_",
    "hpu::normal_",
    "hpu::geometric_",
    "hpu::log_normal_",
    "hpu::exponential_"};

/* below ops modify the o/p dtype in their out of place variant thereby
 * requiring cast node*/
std::unordered_set<std::string> ops_needing_cast = {
    "aten::ge",
    "aten::le",
    "aten::gt",
    "aten::lt"};

void insert_cast_node(
    std::shared_ptr<JitGraph> graph,
    JitNode* node,
    JitNode* insert_after_node,
    const habana::eager::StridedOutInfo& s) {
  torch::jit::WithInsertPoint insert_point(node);
  auto value_in = insert_after_node->output(0);
  auto op_copy = c10::Symbol::fromQualString("aten::_to_copy");
  auto dst_dtype = graph->insertConstant(s.dtype);
  auto dummy_args = graph->insertConstant(torch::jit::IValue());
  auto non_blocking = graph->insertConstant(false);
  auto copy_node = graph->create(
      op_copy,
      {insert_after_node->output(0),
       dst_dtype,
       dummy_args,
       dummy_args,
       dummy_args,
       non_blocking,
       dummy_args},
      1);
  graph->insertNode(copy_node);
  set_deterministic(copy_node);

  insert_after_node->output(0)->replaceAllUsesAfterNodeWith(
      copy_node, copy_node->output(0));
}

void insert_control_edge_node(
    std::shared_ptr<JitGraph> graph,
    JitNode* node,
    JitNode* insert_after_node) {
  torch::jit::WithInsertPoint insert_point(node);

  if (underscored_ops_reported_as_non_inplace.find(
          node->kind().toQualString()) ==
      underscored_ops_reported_as_non_inplace.end()) {
    return;
  }

  auto op_control_edge = c10::Symbol::fromQualString("hpu::control_edge_");
  auto control_edge_node =
      graph->create(op_control_edge, {insert_after_node->output(0)}, 1);
  graph->insertNode(control_edge_node);
  set_deterministic(control_edge_node);

  insert_after_node->output(0)->replaceAllUsesAfterNodeWith(
      control_edge_node, control_edge_node->output(0));
}

JitNode* insert_strided_view_node(
    JitGraph& graph,
    at::Tensor input,
    JitNode* node,
    size_t idx) {
  ViewParam p;
  p.setParam(input);
  torch::jit::WithInsertPoint insert_point(node);

  auto op_strided_view = c10::Symbol::fromQualString("aten::as_strided");
  auto value_sizes = graph.insertConstant(torch::jit::IValue(p.getViewSizes()));
  auto value_strides =
      graph.insertConstant(torch::jit::IValue(p.getViewStrides()));
  auto value_offset =
      graph.insertConstant(torch::jit::IValue(p.getViewOffset()));

  auto value_in = node->input(idx);
  auto jit_node = graph.create(
      op_strided_view, {value_in, value_sizes, value_strides, value_offset}, 1);

  jit_node->output(0)->setType(c10::TensorType::createContiguous(
      input.scalar_type(), input.device(), p.getViewSizes()));

  jit_node->input(0)->setType(c10::TensorType::createContiguous(
      input.scalar_type(), input.device(), input.sizes()));

  set_deterministic(jit_node);
  graph.insertNode(jit_node);

  value_in->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));

  return jit_node;
}

bool check_inplace_op(const EagerOpMetaData& eager_op_meta_data) {
  return (
      (eager_op_meta_data.op_kind_ == habana::eager::eagerOpKind::Inplace) ||
      (eager_op_meta_data.op_kind_ == habana::eager::eagerOpKind::InplaceOut));
}

size_t get_node_output_idx(
    const EagerOpMetaData& op_meta_data,
    JitNode* node,
    size_t idx) {
  int out_idx = -1;
  if (node->outputs().size() == 1) {
    out_idx = 0;
  } else if (op_meta_data.num_out_tensors_ > 1) {
    // the input and output value pointers of out tensors wont match if the out
    // tensors are part of tuple
    out_idx = idx + op_meta_data.num_out_tensors_ - node->inputs().size();
  } else {
    auto value = node->input(idx);
    PT_EAGER_DEBUG("[get_node_output_idx] Search for output value = ", value);
    for (auto i = 0; i < node->outputs().size(); i++) {
      PT_EAGER_DEBUG("[get_node_output_idx] Output value = ", node->output(i));
      if (node->output(i) == value) {
        out_idx = (int)i;
        break;
      }
    }
    HABANA_ASSERT(out_idx != -1, "Invalid node output index!");
  }
  return (size_t)out_idx;
}

bool is_view(const at::Tensor& t) {
  return (habana::is_view_lowering(t) || (!t.is_contiguous()));
}

void collect_output_view_param(
    JitNode* node,
    const std::vector<at::IValue>& inputs,
    const EagerOpMetaData& eager_op_meta_data,
    std::vector<StridedOutInfo>& strided_out_info) {
  auto parse_output_tensor = [&eager_op_meta_data,
                              &inputs,
                              &node,
                              &strided_out_info](
                                 const size_t idx, const at::IValue& out_ival) {
    HABANA_ASSERT(
        out_ival.isTensor(), "Expected tensor input, when parsing idx: ", idx);
    auto output_tensor = out_ival.toTensor();
    if (!is_view(output_tensor)) {
      return;
    }
    HABANA_ASSERT(
        inputs[idx].isTensor(),
        "Tensor lists containing views are unsupported. Failed for idx: ",
        idx);

    StridedOutInfo s;
    auto node_output_idx = get_node_output_idx(eager_op_meta_data, node, idx);
    s.index = node_output_idx;
    s.tensor = output_tensor;
    s.value = node->input(idx);
    s.param = std::make_unique<ViewParam>();
    s.param->setParam(output_tensor);
    s.dtype = output_tensor.scalar_type();
    strided_out_info.emplace_back(std::move(s));
    PT_EAGER_DEBUG(
        "[collect_output_view_param] JIT node outputs count = ",
        node->outputs().size(),
        " idx = ",
        idx,
        " node_output_idx = ",
        node_output_idx);
  };

  for (auto& idx : eager_op_meta_data.out_indices_) {
    if (inputs.at(idx).isScalar() || inputs[idx].isNone()) {
      continue;
    }

    if (inputs[idx].isTensorList()) {
      for (const auto& t : inputs[idx].toTensorVector())
        parse_output_tensor(idx, t);
    } else {
      parse_output_tensor(idx, inputs[idx]);
    }
  }

  size_t inputs_size = inputs.size();
  for (size_t i = inputs_size - eager_op_meta_data.num_out_tensors_;
       i < inputs_size;
       i++) {
    if (inputs[i].isTensorList()) {
      for (const at::Tensor& t : inputs[i].toTensorList())
        parse_output_tensor(i, t);
    } else {
      parse_output_tensor(i, inputs[i]);
    }
  }
}

static JitNode* replace_with_out_of_place_op(
    std::shared_ptr<JitGraph>& graph,
    JitNode* node,
    const EagerOpMetaData& eager_op_meta_data) {
  torch::jit::WithInsertPoint insert_point(node);
  std::string new_kind = eager_op_meta_data.op_name_;

  auto new_node = graph->create(c10::Symbol::fromQualString(new_kind));
  new_node->addInput(node->input(0));
  auto num_inputs = node->inputs().size() - eager_op_meta_data.num_out_tensors_;
  for (size_t i = 1; i < num_inputs; ++i) {
    new_node->addInput(node->input(i));
  }
  new_node->setScope(node->scope());
  new_node->copyAttributes(*node);
  new_node->output(0)->copyMetadata(node->output(0));
  graph->insertNode(new_node);
  node->output(0)->replaceAllUsesWith(new_node->output(0));
  node->destroy();

  return new_node;
}

JitNode* insert_strided_insert_node(
    std::shared_ptr<JitGraph> graph,
    JitNode* node,
    const EagerOpMetaData& eager_op_meta_data,
    StridedOutInfo& s) {
  size_t idx = s.index;
  at::Tensor& output = s.tensor;
  std::unique_ptr<ViewParam>& p = s.param;

  auto op_strided_insert = c10::Symbol::fromQualString("hpu::strided_insert");
  auto value_strides =
      graph->insertConstant(torch::jit::IValue(p->getViewStrides()));
  auto value_offset =
      graph->insertConstant(torch::jit::IValue(p->getViewOffset()));

  auto jit_node = graph->create(
      op_strided_insert,
      {s.value, node->output(idx), value_strides, value_offset},
      1);

  jit_node->input(0)->setType(c10::TensorType::createContiguous(
      output.scalar_type(), output.device(), {p->getTotalElements()}));

  jit_node->input(1)->setType(c10::TensorType::createContiguous(
      output.scalar_type(), output.device(), p->getViewSizes()));

  jit_node->output(0)->setType(c10::TensorType::createContiguous(
      output.scalar_type(), output.device(), {p->getTotalElements()}));

  set_deterministic(jit_node);
  graph->insertNode(jit_node);

  node->output(idx)->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));

  return jit_node;
}

} // namespace

void set_deterministic(JitNode* node) {
  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    auto one = torch::jit::attr::deterministic;
    auto& gconfig = HPURegistrar::get_hpu_global_config();
    node->i_(one, gconfig.getDeterministic());
    PT_EAGER_DEBUG(
        "Deterministic val during Jit Node creation: ", node->i(one));
  }
}

void HandleInputOutputViews(
    std::shared_ptr<JitGraph>& graph,
    const std::vector<at::IValue>& inputs,
    const EagerOpMetaData& eager_op_meta_data,
    bool eager_compiler_supported) {
  PT_EAGER_TRACE;

  PT_EAGER_DEBUG(
      "[HandleInputOutputViews] Eager Op Info = ",
      eager_op_meta_data.to_string());

  JitNode* node{nullptr};

  // We are expecting only single non prim::Constant node in a given graph
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    if (it->kind() != at::prim::Constant &&
        it->kind() != at::prim::ListConstruct) {
      TORCH_CHECK(
          node == nullptr,
          "Expecting exactly one non-const node, but already found ",
          node->kind().toQualString(),
          " and ",
          it->kind().toQualString());
      node = *it;
    }
  }

  // Check if the op type is inplace or inplace-out.
  // If yes, check if output is strided and if so, collect output view param
  // now. This view param will be used in output view handling later.

  PT_EAGER_DEBUG(
      "[HandleInputOutputViews] Op Name: ", node->kind().toQualString());

  PT_EAGER_DEBUG(
      "\nSV node insertion:=====================\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      "[Before]",
      '\n',
      graph->toString(),
      "JIT_IR_Graph_END\n");

  bool is_inplace_op = check_inplace_op(eager_op_meta_data);
  std::vector<StridedOutInfo> strided_out_info;
  if (is_inplace_op) {
    collect_output_view_param(
        node, inputs, eager_op_meta_data, strided_out_info);
  }

  int num_inputs = node->inputs().size();
  std::vector<JitNode*> strided_view_nodes;
  strided_view_nodes.reserve(num_inputs);

  auto insert_stride_if_view = [&strided_view_nodes, &graph](
                                   at::Tensor& input_tensor,
                                   JitNode* node,
                                   size_t idx) {
    if (input_tensor.device().type() != c10::DeviceType::HPU) {
      return;
    }

    if (!is_view(input_tensor)) {
      return;
    }

    auto jit_node = insert_strided_view_node(*graph, input_tensor, node, idx);
    strided_view_nodes.push_back(jit_node);
  };

  for (auto idx = 0u; idx < num_inputs; ++idx) {
    auto val = inputs.at(idx);
    if (val.isTensor()) {
      auto& t = val.toTensor();
      insert_stride_if_view(t, node, idx);
    } else if (val.isTensorList()) {
      JitNode* list_node = node->input(idx)->node();
      size_t li = 0;
      for (auto& t : val.toTensorVector()) {
        insert_stride_if_view(t, list_node, li++);
      }
    }
  }

  if (strided_view_nodes.size()) {
    PT_EAGER_DEBUG(
        "\nSV node insertion:=====================\n",
        "JIT_IR_Graph_BEGIN\n",
        "Graph ",
        "[After]",
        '\n',
        graph->toString(),
        "JIT_IR_Graph_END\n");
  } else {
    PT_EAGER_DEBUG("\nSV node insertion not required.=====================\n");
  }

  // Insert control edge for nodes that require it when graph compiler is used
  // (i.e. Gaudi)
  if (!eager_compiler_supported && strided_view_nodes.size()) {
    auto last_node = strided_view_nodes.back();
    insert_control_edge_node(graph, node, last_node);
  }

  if (!is_inplace_op or strided_out_info.empty()) {
    return;
  }

  // We reach this point if the op type is inplace or inplace-out with strided
  // output. Note, output view param is already collected.

  JitNode* new_node = node;
  // These kernels completely ignore the data in input tensor and hence the
  // input tensor can be reused by updating inplace. Further it also avoids
  // implementing out of place variants
  if (!underscored_ops_reported_as_non_inplace.count(
          node->kind().toQualString())) {
    if (eager_op_meta_data.num_out_tensors_ <= 1)
      new_node = replace_with_out_of_place_op(graph, node, eager_op_meta_data);
  }

  PT_EAGER_DEBUG(
      "\nSI node insertion:=====================\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      "[Before]",
      '\n',
      graph->toString(),
      "JIT_IR_Graph_END\n");

  for (size_t idx = 0; idx < strided_out_info.size(); idx++) {
    auto si_node = insert_strided_insert_node(
        graph, new_node, eager_op_meta_data, strided_out_info.at(idx));

    if (ops_needing_cast.find(new_node->kind().toQualString()) ==
        ops_needing_cast.end())
      continue;

    insert_cast_node(graph, si_node, new_node, strided_out_info.at(idx));
  }

  PT_EAGER_DEBUG(
      "\nSI node insertion:=====================\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      "[After]",
      '\n',
      graph->toString(),
      "JIT_IR_Graph_END\n");
}

} // namespace eager
} // namespace habana
