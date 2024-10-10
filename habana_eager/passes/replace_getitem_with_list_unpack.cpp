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

#include <c10/util/ArrayRef.h>

#include "habana_eager/graph_exec.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

struct ListUnpackDesc {
  explicit ListUnpackDesc(torch::jit::Node* list_unpack_node)
      : m_node(list_unpack_node) {}

  void add_output(size_t idx, torch::jit::Value* out) {
    if (m_output_list.size() < idx + 1) {
      m_output_list.resize(idx + 1);
    }
    HABANA_ASSERT(nullptr == m_output_list[idx]);
    m_output_list[idx] = out;
  }

  void update_node_outputs() {
    HABANA_ASSERT(m_output_list.size() > 0);
    for (torch::jit::Value* out : m_output_list) {
      if (nullptr != out) {
        auto new_out = m_node->addOutput()->copyMetadata(out);
        out->replaceAllUsesWith(new_out);
      } else {
        m_node->addOutput();
      }
    }
  }

  std::vector<torch::jit::Value*> m_output_list;
  torch::jit::Node* m_node;
};

struct ReplaceGetItemWithListUnpackPass {
  explicit ReplaceGetItemWithListUnpackPass(
      std::shared_ptr<torch::jit::Graph> graph)
      : m_graph(std::move(graph)) {}
  bool run() {
    bool changed{processBlocks(m_graph->block())};
    return changed;
  }

 private:
  bool processBlocks(at::ArrayRef<torch::jit::Block*> blocks) {
    bool changed{false};
    for (auto block : blocks) {
      changed |= processBlock(block);
    }
    return changed;
  }

  bool processBlock(torch::jit::Block* block) {
    static const auto getitem_symbol{
        c10::Symbol::fromQualString("aten::__getitem__")};
    static const auto constant_symbol{
        c10::Symbol::fromQualString("prim::Constant")};
    static const auto list_unpack_symbol{
        c10::Symbol::fromQualString("prim::ListUnpack")};

    std::map<torch::jit::Value*, ListUnpackDesc> list_unpack_desc_map;
    std::map<torch::jit::Value*, torch::jit::Node*> output_to_const_map;
    std::set<torch::jit::Node*> nodes_to_remove;

    bool changed{false};
    // First step is collecting all prim::Constatnt in block.
    // Const values are used as indexes in __getitem__
    // All const nodes are mapped by output for further usage.
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      if (constant_symbol == it->kind()) {
        torch::jit::Node* node{*it};
        HABANA_ASSERT(1 == node->outputs().size());
        output_to_const_map[node->output(0)] = node;
      }
    }

    // Second pass: find all  aten::__getitem__.
    // Results of this search are aggregated by first getitem input - which
    // should be TensorList. For every input we create structure descripting new
    // ListUnpack node that will replace all aten::__getitem__ nodes.
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      if (getitem_symbol == it->kind()) {
        torch::jit::Node* node{*it};
        HABANA_ASSERT(2 == node->inputs().size());
        auto list_unpack_input{node->input(0)};
        HABANA_ASSERT(
            *list_unpack_input->type() == *torch::ListType::ofTensors(),
            "Unsupported aten::__getitem__ in compiled graph");
        if (list_unpack_desc_map.find(list_unpack_input) ==
            list_unpack_desc_map.end()) {
          torch::jit::WithInsertPoint insert_guard{node};
          auto graph{node->owningGraph()};
          auto list_unpack_node{
              graph->insertNode(graph->create(list_unpack_symbol, 0))};
          list_unpack_node->addInput(list_unpack_input);
          list_unpack_desc_map.emplace(std::make_pair(
              list_unpack_input, ListUnpackDesc(list_unpack_node)));
        }

        ListUnpackDesc& desc{
            list_unpack_desc_map.find(list_unpack_input)->second};
        // Obtaining output position in ListUnpack from output nodes gattherd
        // in previous loop.
        auto getitem_idx_input{node->input(1)};
        HABANA_ASSERT(
            output_to_const_map.find(getitem_idx_input) !=
                output_to_const_map.end(),
            "Unable to determine input index");
        torch::jit::Node* const_node_with_idx{
            output_to_const_map.at(getitem_idx_input)};
        static const auto value_attr{torch::jit::Symbol::attr("value")};
        long out_idx{const_node_with_idx->i(value_attr)};
        // Adding information about output value and index on list to descriptor
        desc.add_output(out_idx, node->output(0));
        // Collecting which nodes should be removed at the end.
        nodes_to_remove.insert(node);
        nodes_to_remove.insert(const_node_with_idx);
        node->removeAllInputs();
        changed |= true;
      }
    }

    // For every descriptor we completed we will need to update output of
    // underlying op. At this moment we should have all index ready to be
    // filled.
    for (auto& it : list_unpack_desc_map) {
      ListUnpackDesc& desc{it.second};
      desc.update_node_outputs();
      changed |= true;
    }

    // Last pass: remove all nodes that are no longer necessary. Some
    // prim::Constant nodes might remain if there are still used elsewhere in
    // graph
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      torch::jit::Node* node{*it};
      if (nodes_to_remove.end() != nodes_to_remove.find(node)) {
        HABANA_ASSERT(node->outputs().size() == 1);
        if (!node->output(0)->hasUses()) {
          it.destroyCurrent();
          changed |= true;
        }
      }
    }

    return changed;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
};

bool ReplaceGetItemWithListUnpack(std::shared_ptr<torch::jit::Graph> graph) {
  PT_EAGER_TRACE;
  ReplaceGetItemWithListUnpackPass pass{graph};
  bool changed{pass.run()};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
  return changed;
}

} // namespace pass
} // namespace graph
} // namespace habana
