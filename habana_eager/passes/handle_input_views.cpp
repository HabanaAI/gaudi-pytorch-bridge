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

  std::map<int64_t, std::vector<int64_t>> get_base_sizes_to_set_during_launch() {
    return m_input_base_sizes_to_set;
  }

 private:
  bool processInputs(
      at::ArrayRef<torch::jit::Value*> inputs,
      torch::jit::Stack& example_inputs) {
    bool changed{false};
    for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
      torch::jit::Value* input{inputs.at(input_idx)};
      if (!example_inputs[input_idx].isTensor()) {
        continue;
      }
      torch::Tensor input_tensor{example_inputs[input_idx].toTensor()};
      auto storage_meta{habana::get_storage_extra_meta(input_tensor)};

      if (habana::is_view_lowering(input_tensor) ||
          !input_tensor.is_contiguous()) {
        auto& first_use{input->uses()[0]};
        torch::jit::Node* user{first_use.user};

        static const std::set<c10::Symbol> view_ops_symbols{
            c10::Symbol::fromQualString("aten::as_strided"),
            c10::Symbol::fromQualString("aten::slice_scatter"),
            c10::Symbol::fromQualString("aten::select_scatter"),
            c10::Symbol::fromQualString("aten::as_strided_scatter")};

        if (view_ops_symbols.find(user->kind()) != view_ops_symbols.end()) {
          // Moving for next input as view for this one are already handled in
          // graph
          continue;
        }
        auto view_params{std::make_unique<habana::eager::ViewParam>()};
        view_params->setParam(input_tensor);
        m_input_base_sizes_to_set[input_idx] = std::vector<int64_t>();
        insert_strided_view_node(
            input_tensor,
            storage_meta,
            user,
            input,
            view_params,
            m_input_base_sizes_to_set.at(input_idx));
        changed |= true;
      }
    }

    return changed;
  }

  void insert_strided_view_node(
      at::Tensor input_tensor,
      habana::StorageExtraMeta* input_smeta,
      torch::jit::Node* node,
      torch::jit::Value* value_in,
      std::unique_ptr<habana::eager::ViewParam>& p,
      std::vector<int64_t>& base_sizes_to_set) {
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

    base_sizes_to_set = habana::get_base_tensor_size(input_tensor);

    jit_node->input(0)->setType(c10::TensorType::createContiguous(
        input_tensor.scalar_type(),
        input_tensor.device(),
        input_tensor.sizes()));

    m_graph->insertNode(jit_node);

    value_in->replaceAllUsesAfterNodeWith(jit_node, jit_node->output(0));
  }

 private:
  std::shared_ptr<torch::jit::Graph> m_graph;
  std::map<int64_t, std::vector<int64_t>> m_input_base_sizes_to_set;
};

void HandleInputViews(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& example_inputs,
    std::map<int64_t, std::vector<int64_t>>& input_base_sizes_map) {
  PT_EAGER_TRACE;
  HandleInputViewsPass pass{graph};
  bool changed{pass.run(example_inputs)};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
  input_base_sizes_map = pass.get_base_sizes_to_set_during_launch();
}

} // namespace pass
} // namespace graph
} // namespace habana