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
    PT_BRIDGE_DEBUG(
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
  impl->set_sizes_and_strides(
      at::IntArrayRef{p->getTotalElements()}, at::IntArrayRef{1});
  jit_node->input(0)->setType(c10::TensorType::createContiguous(
      input.scalar_type(), input.device(), input.sizes()));

  set_deterministic(jit_node);
  graph->insertNode(jit_node);

  value_in->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));

  return jit_node;
}

void HandleInputOutputViews(
    std::shared_ptr<JitGraph>& graph,
    const SmallTensorVector& inputs) {
  PT_EAGER_TRACE;

  JitNode* node{nullptr};

  // only single-node graph is assumed
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    if (it->kind() == at::prim::Constant)
      continue;
    else {
      TORCH_CHECK(
          node == nullptr,
          "Expecting exactly one non-const node, but already found ",
          node->kind(),
          " and ",
          it->kind());
      node = *it;
    }
  }

  PT_BRIDGE_DEBUG(
      "\nBefore SV node insertion:=====================\n",
      "JIT_IR_Graph_BEGIN\n",
      "Graph ",
      "[Before Pass]",
      '\n',
      graph->toString(),
      "JIT_IR_Graph_END\n");

  auto strided_view_node = 0;
  for (size_t idx = 0; idx < inputs.size(); idx++) {
    std::unique_ptr<ViewParam> p = std::make_unique<ViewParam>();
    at::Tensor input_tensor = inputs.at(idx);
    if (!input_tensor.is_contiguous()) {
      p->setParam(input_tensor);
      insert_strided_view_node(graph, input_tensor, node, idx, p);
      strided_view_node++;
    }
  }

  if (strided_view_node > 0) {
    PT_BRIDGE_DEBUG(
        "\nAfter SV node insertion:=====================\n",
        "JIT_IR_Graph_BEGIN\n",
        "Graph ",
        "[After Pass]",
        '\n',
        graph->toString(),
        "JIT_IR_Graph_END\n");
  } else {
    PT_BRIDGE_DEBUG("\nSV node insertion not required.=====================\n");
  }
}

} // namespace eager
} // namespace habana
