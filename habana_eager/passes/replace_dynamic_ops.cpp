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

#include <c10/util/ArrayRef.h>

#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_dynamic_ops.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {
#define PT_MAX_SHAPETENSOR_INPUT 10

struct HandleDynamicOpsPass {
  explicit HandleDynamicOpsPass(
      std::shared_ptr<torch::jit::Graph> graph,
      std::shared_ptr<DynamicGraphMetaData> dmeta,
      std::map<int64_t, std::vector<int64_t>>* input_new_base_sizes)
      : m_graph(std::move(graph)), m_dmeta(std::move(dmeta)) {
    m_input_new_base_sizes = input_new_base_sizes;
  }

  bool run(torch::jit::Stack& stack) {
    propagateShape(stack);
    bool changed{processBlocks(m_graph->block(), stack)};
    return changed;
  }

 private:
  void createGraphInputStackIndexMap(GraphInputIndexMap& org_stack_index_map) {
    for (int idx = 0; idx < m_graph->inputs().size(); idx++) {
      auto input = m_graph->inputs().at(idx);
      auto name = input->debugName();
      org_stack_index_map[name] = idx;
    }
  }

  void eliminateUnusedInputs(torch::jit::Block* block) {
    c10::ArrayRef<torch::jit::Value*> inputs = block->inputs();
    size_t i = inputs.size() - 1;
    for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
      torch::jit::Value* input = *it;
      if (!input->hasUses()) {
        std::string inputName = input->debugName();
        PT_EAGER_DEBUG("Removing unused Input = ", inputName);
        block->eraseInput(i);
        m_dmeta->remove_input_indexes.push_back(i);
      }
      i--;
    }
  }

  void handlePrimConstantNode(torch::jit::Node* node) {
    auto node_vals = node->outputs();
    for (const auto value : node_vals) {
      torch::jit::IValue const_ivalue = toIValue(value).value();
      m_value_ivalue_map[value] = std::make_shared<IVal>(const_ivalue);
    }
  }

  void handlePrimListConstructNode(torch::jit::Node* node) {
    const auto& node_ins = node->inputs();
    auto node_vals = node->outputs();
    HABANA_ASSERT(node_vals.size() == 1);
    IValPtrShared ival =
        GetPrimListConstructNodeOuputIValue(node, m_value_ivalue_map);
    m_value_ivalue_map[node_vals[0]] = ival;
  }

  void handlePrimListUnpackNode(torch::jit::Node* node) {
    auto node_vals = node->outputs();
    for (const auto& input : node->inputs()) {
      const auto& name = input->debugName();
      auto tensors = (*m_value_ivalue_map[input]).toTensorList();
      for (int i = 0; i < tensors.size(); ++i) {
        const at::Tensor& tensor = tensors[i];
        m_value_ivalue_map[node_vals[i]] = std::make_shared<IVal>(tensor);
      }
    }
  }

  void dumpValueIValueMap() {
    PT_EAGER_DEBUG("Map m_value_ivalue_map size :", m_value_ivalue_map.size());
    for (auto it : m_value_ivalue_map) {
      if (it.second->isTensor()) {
        PT_EAGER_DEBUG(
            "value name = ",
            it.first->debugName(),
            ", sizes = ",
            it.second->toTensor().sizes());
      } else {
        PT_EAGER_DEBUG("value name = ", it.first->debugName())
      }
    }
  }

  void propagateShape(torch::jit::Stack& org_stack) {
    // Run SIF
    std::unordered_map<CValPtr, torch::jit::IValue> value_ivalue_map;
    HabanaLaunchOpPT::RunHybridSif(m_graph, org_stack, value_ivalue_map);
    for (auto val_ivalue : value_ivalue_map) {
      m_value_ivalue_map[val_ivalue.first] =
          std::make_shared<IVal>(val_ivalue.second);
    }
    // dump shapes
    dumpValueIValueMap();
  }

  bool processBlock(torch::jit::Block* block, torch::jit::Stack& org_stack) {
    bool changed{false};
    GraphInputIndexMap org_stack_index_map;
    createGraphInputStackIndexMap(org_stack_index_map);
    HABANA_ASSERT(m_graph->inputs().size() == org_stack.size());

    // First Pass: Repace all dynamic shape ops with hpu implementation.
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      std::string node_name = it->kind().toQualString();
      torch::jit::Node* node{*it};
      DynamicOpPtr dsOp = DSOpsRegistry().get(node_name);
      if (!dsOp)
        continue;
      PT_EAGER_DEBUG("Replace dynamic Op: ", node_name);
      dsOp->m_input_new_base_sizes = m_input_new_base_sizes;
      changed = dsOp->ReplaceWithDynamicHPUOp(
          node, org_stack, org_stack_index_map, m_value_ivalue_map, m_dmeta);
    }

    // Second pass: remove all nodes that are no longer necessary.
    torch::jit::EliminateDeadCode(m_graph);

    //  Last pass: Remove all the unused graph inputs as well.
    eliminateUnusedInputs(block);
    return changed;
  }

  bool processBlocks(
      at::ArrayRef<torch::jit::Block*> blocks,
      torch::jit::Stack& org_stack) {
    bool changed{false};
    for (auto block : blocks) {
      changed |= processBlock(block, org_stack);
    }

    return changed;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
  std::shared_ptr<DynamicGraphMetaData> m_dmeta;
  CValuePtrToIValuePtrMap m_value_ivalue_map;
  std::map<int64_t, std::vector<int64_t>>* m_input_new_base_sizes;
};

void HandleDynamicOps(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    std::map<int64_t, std::vector<int64_t>>* input_new_base_sizes) {
  PT_EAGER_TRACE;
  HandleDynamicOpsPass pass{graph, dmeta, input_new_base_sizes};

  bool changed{pass.run(stack)};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
  }
}

void HandlePostDynamic(
    std::shared_ptr<DynamicGraphMetaData> dgraph_meta,
    std::map<int64_t, std::vector<int64_t>>& input_base_sizes_map) {
  // 1. Correct the input indexes in input_base_sizes_map
  std::map<int64_t, int64_t> key_map;
  for (auto& input_datasize_pair : input_base_sizes_map) {
    int key = input_datasize_pair.first;

    // Check if key is greater than any element in the vector
    int new_key = key;
    for (int idx : dgraph_meta->remove_input_indexes) {
      if (key > idx) {
        --new_key;
        key_map[key] = new_key;
      }
    }
  }
  size_t erase_count = 0;
  for (auto key_pair : key_map) {
    auto value = input_base_sizes_map[key_pair.first];
    // Reduce the key by 1
    int newKey = key_pair.second;
    input_base_sizes_map[newKey] = value;
    input_base_sizes_map.erase(key_pair.first);
  }
}

void ResolveNegativeSTSizes(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dmeta) {
  PT_EAGER_TRACE;
  std::unordered_map<CValPtr, torch::jit::IValue> m_value_ivalue_map;
  HabanaLaunchOpPT::RunHybridSif(graph, stack, m_value_ivalue_map);

  for (auto it = dmeta->negative_size_nodes.begin();
       it != dmeta->negative_size_nodes.end();
       it++) {
    torch::jit::Node* node{*it};
    std::string node_name = node->kind().toQualString();
    DynamicOpPtr dsOp = DSOpsRegistry().get(node_name);
    if (!dsOp)
      continue;
    dsOp->ResolveNegativeSizes(graph, stack, node, m_value_ivalue_map);
  }
}

void HandleDynamicInputPatching(
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    bool is_first_launch) {
  PT_EAGER_TRACE;

  PT_EAGER_DEBUG(
      "Number of dynamic input to be patched:",
      dmeta->ds_input_patching_list.size());
  // Combine the original input stack and dynamic stack created at the runtime
  // into a single stack.
  c10::SmallVector<torch::jit::IValue*, PT_MAX_SHAPETENSOR_INPUT> dtensor_list;
  c10::SmallVector<habana::graph::SymIntData, PT_MAX_SHAPETENSOR_INPUT>
      scalar_list;
  c10::SmallVector<std::vector<int64_t>, PT_MAX_SHAPETENSOR_INPUT> tensor_list;
  for (auto dtensor_info : dmeta->ds_input_patching_list) {
    auto dtensor_indexes = dtensor_info.second;
    dtensor_list.clear();
    scalar_list.clear();
    for (auto it : dtensor_indexes) {
      stack.emplace_back(dmeta->ds_stack[it]);
      dtensor_list.emplace_back(&(dmeta->ds_stack[it]));
      scalar_list.emplace_back(dmeta->ds_tensor_to_scalar_map[it]);
      tensor_list.emplace_back(dmeta->ds_tensor_to_tensor_map[it]);
    }

    if (!is_first_launch) {
      dtensor_info.first(dtensor_list, scalar_list, tensor_list, stack);
    }
  }

  // We want to remove elements from stack in reverse order so that
  // the indexes for others don't get messed up
  auto stack_begin = stack.begin();
  for (auto idx : dmeta->remove_input_indexes) {
    stack.erase(stack_begin + idx);
  }
}

} // namespace pass
} // namespace graph
} // namespace habana
