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
 *******************************************************************************/

// #include <c10/util/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <queue>

#include "habana_eager/graph_exec.h"

#include "habana_eager/eager_view.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

struct HandleInputViewsPass {
  explicit HandleInputViewsPass(std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}

  bool run(torch::jit::Stack& example_inputs) {
    bool changed{processInputs(m_graph->inputs(), example_inputs)};
    return changed;
  }

 private:
  bool processInputs(
      at::ArrayRef<torch::jit::Value*> inputs,
      torch::jit::Stack& example_inputs) {
    bool changed{false};
    for (int input_idx = 0; input_idx < inputs.size(); input_idx++) {
      torch::jit::Value* input{inputs.at(input_idx)};
      if (!example_inputs[input_idx].isTensor()) {
        continue;
      }
      torch::Tensor input_tensor{example_inputs[input_idx].toTensor()};
      auto tensor_meta{habana::get_tensor_extra_meta(input_tensor)};

      if (tensor_meta->is_view_lowering() || !input_tensor.is_contiguous()) {
        auto& first_use{input->uses()[0]};
        torch::jit::Node* user{first_use.user};
        auto view_params{std::make_unique<habana::eager::ViewParam>()};
        view_params->setParam(input_tensor);
        insert_strided_view_node(
            input_tensor, tensor_meta, user, input, view_params);
        changed |= true;
      }
    }

    return changed;
  }

  void insert_strided_view_node(
      at::Tensor input_tensor,
      habana::TensorExtraMeta* input_tmeta,
      torch::jit::Node* node,
      torch::jit::Value* value_in,
      std::unique_ptr<habana::eager::ViewParam>& p) {
    PT_EAGER_TRACE;
    torch::jit::WithInsertPoint insert_point(node);

    auto op_strided_view = c10::Symbol::fromQualString("aten::as_strided");
    auto value_sizes =
        m_graph->insertConstant(torch::jit::IValue(p->getViewSizes()));
    auto value_strides =
        m_graph->insertConstant(torch::jit::IValue(p->getViewStrides()));
    auto value_offset =
        m_graph->insertConstant(torch::jit::IValue(p->getViewOffset()));

    auto jit_node = m_graph->create(
        op_strided_view,
        {value_in, value_sizes, value_strides, value_offset},
        1);

    jit_node->output(0)->setType(c10::TensorType::createContiguous(
        input_tensor.scalar_type(), input_tensor.device(), p->getViewSizes()));

    std::vector<int64_t> base_sizes;
    if (input_tmeta->get_memory_permutation().size()) {
      base_sizes = input_tmeta->get_base_tensor_size();
    } else {
      base_sizes = {p->getTotalElements()};
    }

    auto* impl = input_tensor.unsafeGetTensorImpl();
    impl->set_storage_offset(0);
    impl->set_sizes_contiguous(base_sizes);

    jit_node->input(0)->setType(c10::TensorType::createContiguous(
        input_tensor.scalar_type(),
        input_tensor.device(),
        input_tensor.sizes()));

    m_graph->insertNode(jit_node);

    value_in->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));
  }

 private:
  std::shared_ptr<torch::jit::Graph> m_graph;
};

void HandleInputViews(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs) {
  PT_EAGER_TRACE;
  HandleInputViewsPass pass{graph};
  bool changed{pass.run(example_inputs)};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
}

} // namespace pass
} // namespace graph
} // namespace habana