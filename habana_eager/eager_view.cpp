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

static void set_deterministic(JitNode* node) {
  if (GET_ENV_FLAG_NEW(PT_HPU_DETERMINISTIC_ENABLE)) {
    auto one = torch::jit::attr::alpha;
    auto& device = synapse_helpers::HPURegistrar::get_device();
    node->i_(one, device.getDeterministic());
    PT_EAGER_DEBUG(
        "Deterministic val during Jit Node creation: ", node->i(one));
  }
}

static JitNode* insert_strided_view_node(
    std::shared_ptr<JitGraph> graph,
    at::Tensor input,
    JitNode* node,
    size_t idx,
    std::unique_ptr<ViewParam>& p) {
  torch::jit::WithInsertPoint insert_point(node);

  auto op_strided_view = c10::Symbol::fromQualString("aten::as_strided");
  auto value_sizes =
      graph->insertConstant(torch::jit::IValue(p->getViewSizes()));
  auto value_strides =
      graph->insertConstant(torch::jit::IValue(p->getViewStrides()));
  auto value_offset =
      graph->insertConstant(torch::jit::IValue(p->getViewOffset()));

  auto value_in = node->input(idx);
  auto jit_node = graph->create(
      op_strided_view, {value_in, value_sizes, value_strides, value_offset}, 1);

  jit_node->output(0)->setType(c10::TensorType::createContiguous(
      input.scalar_type(), input.device(), p->getViewSizes()));

  auto* impl = input.unsafeGetTensorImpl();
  impl->set_storage_offset(0);
  std::vector<int64_t> base_sizes;
  auto input_tmeta{habana::get_tensor_extra_meta(input)};
  if (input_tmeta->get_memory_permutation().size()) {
    base_sizes = input_tmeta->get_base_tensor_size();
  } else {
    base_sizes = {p->getTotalElements()};
  }
  impl->set_sizes_contiguous(base_sizes);

  jit_node->input(0)->setType(c10::TensorType::createContiguous(
      input.scalar_type(), input.device(), input.sizes()));

  set_deterministic(jit_node);
  graph->insertNode(jit_node);

  value_in->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));

  return jit_node;
}

static bool check_inplace_op(const EagerOpMetaData& eager_op_meta_data) {
  return (
      (eager_op_meta_data.op_kind_ == habana::eager::eagerOpKind::Inplace) ||
      (eager_op_meta_data.op_kind_ == habana::eager::eagerOpKind::InplaceOut));
}

static std::unordered_set<size_t> get_input_tensors_positions(
    JitNode* node,
    const std::vector<at::IValue>& inputs,
    const EagerOpMetaData& eager_op_meta_data) {
  PT_EAGER_DEBUG("JIT node inputs count : ", node->inputs().size());
  PT_EAGER_DEBUG("JIT node outputs count : ", node->outputs().size());
  PT_EAGER_DEBUG("Total inputs : ", inputs.size());

  auto& out_indices = eager_op_meta_data.out_indices_;
  if (eager_op_meta_data.op_kind_ == InplaceOut) {
    HABANA_ASSERT(!out_indices.empty());
    PT_EAGER_DEBUG("Total output tensors : ", out_indices.size());
    HABANA_ASSERT(inputs.size() >= out_indices.size());
  }

  std::unordered_set<size_t> in_indices;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto val = inputs[i];
    if (!val.isTensor()) {
      continue;
    }

    switch (eager_op_meta_data.op_kind_) {
      case InplaceOut:
        if (!out_indices.count(i))
          in_indices.insert(i);
        break;
      case Inplace:
      case OutOfPlace:
      default:
        in_indices.insert(i);
        break;
    }
  }

  return in_indices;
}

static size_t get_node_output_idx(JitNode* node, size_t idx) {
  if (node->outputs().size() == 1) {
    return 0;
  } else {
    int out_idx = -1;
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
    return (size_t)out_idx;
  }
}

static void collect_output_view_param(
    JitNode* node,
    const std::vector<at::IValue>& inputs,
    const EagerOpMetaData& eager_op_meta_data,
    std::vector<StridedOutInfo>& strided_out_info) {
  if (eager_op_meta_data.out_indices_.empty()) {
    return;
  }

  auto parse_output_tensor = [&inputs, &node, &strided_out_info](
                                 const size_t idx, const at::IValue& out_ival) {
    HABANA_ASSERT(
        out_ival.isTensor(), "Expected tensor input, when parsing idx: ", idx);
    auto output_tensor = out_ival.toTensor();
    auto output_tmeta{habana::get_tensor_extra_meta(output_tensor)};
    if (output_tmeta->is_view_lowering() || !output_tensor.is_contiguous()) {
      HABANA_ASSERT(
          inputs[idx].isTensor(),
          "Tensor lists containing views are unsupported. Failed for idx: ",
          idx);
      StridedOutInfo s;
      auto node_output_idx = get_node_output_idx(node, idx);
      s.index = node_output_idx;
      s.tensor = output_tensor;
      s.value = node->input(idx);
      s.param = std::make_unique<ViewParam>();
      s.param->setParam(output_tensor);
      strided_out_info.emplace_back(std::move(s));
      PT_EAGER_DEBUG(
          "[collect_output_view_param] JIT node outputs count = ",
          node->outputs().size(),
          "idx = ",
          idx,
          "node_output_idx = ",
          node_output_idx);
    }
  };

  for (auto& idx : eager_op_meta_data.out_indices_) {
    if (inputs.at(idx).isScalar() || inputs[idx].isNone()) {
      continue;
    }

    if (inputs[idx].isList()) {
      for (const auto& list_element : inputs[idx].toList())
        parse_output_tensor(idx, list_element.get());
    } else {
      parse_output_tensor(idx, inputs[idx]);
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
  auto num_inputs = node->inputs().size();
  if (eager_op_meta_data.op_kind_ == habana::eager::eagerOpKind::InplaceOut) {
    num_inputs -= eager_op_meta_data.out_indices_.size();
  }
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

static JitNode* insert_strided_insert_node(
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

void HandleInputOutputViews(
    std::shared_ptr<JitGraph>& graph,
    const std::vector<at::IValue>& inputs,
    const EagerOpMetaData& eager_op_meta_data) {
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
  bool is_inplace_op = check_inplace_op(eager_op_meta_data);
  std::vector<StridedOutInfo> strided_out_info;
  if (is_inplace_op) {
    collect_output_view_param(
        node, inputs, eager_op_meta_data, strided_out_info);
  }

  PT_EAGER_DEBUG(
      "\nSV node insertion:=====================\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      "[Before]",
      '\n',
      graph->toString(),
      "JIT_IR_Graph_END\n");

  auto strided_view_node = 0;
  auto input_tensor_pos =
      get_input_tensors_positions(node, inputs, eager_op_meta_data);
  for (auto& idx : input_tensor_pos) {
    auto val = inputs.at(idx);
    HABANA_ASSERT(val.isTensor(), "Non-tensor value");
    auto input_tensor = val.toTensor();
    auto input_tmeta{habana::get_tensor_extra_meta(input_tensor)};

    if (input_tmeta->is_view_lowering() || !input_tensor.is_contiguous()) {
      std::unique_ptr<ViewParam> p_in = std::make_unique<ViewParam>();
      p_in->setParam(input_tensor);
      insert_strided_view_node(graph, input_tensor, node, idx, p_in);
      strided_view_node++;
    }
  }

  if (strided_view_node > 0) {
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

  if (is_inplace_op) {
    if (strided_out_info.empty()) {
      return;
    }

    // We reach this point if the op type is inplace or inplace-out with strided
    // output. Note, output view param is already collected.

    JitNode* new_node =
        replace_with_out_of_place_op(graph, node, eager_op_meta_data);

    PT_EAGER_DEBUG(
        "\nSI node insertion:=====================\n",
        "JIT_IR_Graph_BEGIN\n",
        "Graph ",
        "[Before]",
        '\n',
        graph->toString(),
        "JIT_IR_Graph_END\n");

    for (size_t idx = 0; idx < strided_out_info.size(); idx++) {
      insert_strided_insert_node(
          graph, new_node, eager_op_meta_data, strided_out_info.at(idx));
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
}

} // namespace eager
} // namespace habana
