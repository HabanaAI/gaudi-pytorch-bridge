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

#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/graph_dynamic.h"
#include "habana_eager/graph_dynamic_ops.h"
#include "habana_helpers/logging_pt.h"

namespace habana {
namespace graph {
namespace pass {

using IVal = torch::jit::IValue;
using IValPtrShared = std::shared_ptr<IVal>;
using CValPtr = const torch::jit::Value*;

struct HandleDynamicOpsPass {
  explicit HandleDynamicOpsPass(
      std::shared_ptr<torch::jit::Graph> graph,
      std::shared_ptr<DynamicGraphMetaData> dmeta)
      : m_graph(std::move(graph)), m_dmeta(std::move(dmeta)) {}

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

  void executeNode(const torch::jit::Node* node, torch::jit::Stack& inputs) {
    try {
      torch::jit::Operator jit_op = node->getOperator();
      jit_op.getOperation()(inputs);
    } catch (std::exception& e) {
      // catch runtime error due to non-implmentation/mismatch
      PT_EAGER_DEBUG("Catch Exception in DS executeNode ", e.what());
      auto input_tensor = inputs[0].toTensor();
      torch::jit::drop(inputs, node->inputs().size());
      auto node_outs = node->outputs();
      for (int i = 0; i < node_outs.size(); i++) {
        auto output_val = node_outs[i];
        if (output_val->type()->kind() == c10::TypeKind::TensorType) {
          auto result = input_tensor.clone();
          torch::jit::pack(inputs, std::move(result));
        }
      }
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

  void propagateShape(const torch::jit::Stack& org_stack) {
    // Set the shapes of input tensors in the map
    const auto& g_inputs = m_graph->inputs();
    for (size_t i = 0; i < org_stack.size(); ++i) {
      const auto& name = g_inputs[i]->debugName();
      m_value_ivalue_map[g_inputs[i]] = std::make_shared<IVal>(org_stack[i]);
    }

    // Propagate the shapes through the graph nodes
    for (const auto& node : m_graph->nodes()) {
      torch::jit::Stack nodeInputs;
      if (node->kind() == torch::jit::prim::Constant) {
        handlePrimConstantNode(node);
        continue;
      } else if (node->kind() == torch::jit::prim::ListConstruct) {
        handlePrimListConstructNode(node);
        continue;
      } else if (node->kind() == torch::jit::prim::ListUnpack) {
        handlePrimListUnpackNode(node);
        continue;
      }

      for (const auto& input : node->inputs()) {
        const auto& name = input->debugName();
        if ((*m_value_ivalue_map[input]).isTensor()) {
          auto input_ct = (*m_value_ivalue_map[input]).toTensor().to(c10::kCPU);
          nodeInputs.push_back(torch::jit::IValue(input_ct));
        } else {
          nodeInputs.push_back(*m_value_ivalue_map[input]);
        }
      }

      executeNode(node, nodeInputs);

      auto node_outs = node->outputs();
      auto outputIValues = torch::jit::last(nodeInputs, node_outs.size());
      for (int i = 0; i < outputIValues.size(); i++) {
        auto output_val = node_outs[i];
        auto output_ival = outputIValues[i];
        m_value_ivalue_map[output_val] = std::make_shared<IVal>(output_ival);
      }
    }

    dumpValueIValueMap();
  }

  std::vector<at::Tensor> getInputTensers(const torch::jit::Node* node) {
    std::vector<at::Tensor> in_tensors;
    for (const auto& input : node->inputs()) {
      PT_EAGER_DEBUG("Node input name = ", input->debugName());
      auto ivalue = m_value_ivalue_map[const_cast<torch::jit::Value*>(input)];
      HABANA_ASSERT(
          ivalue != nullptr,
          "Node = ",
          node->kind().toQualString(),
          ", input = ",
          input->debugName(),
          " not found in m_value_ivalue_map!!");
      if (ivalue->isTensor()) {
        in_tensors.push_back(ivalue->toTensor());
      }
    }
    return in_tensors;
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
      std::vector<at::Tensor> in_tensors = getInputTensers(node);
      bool changed = dsOp->ReplaceWithDynamicHPUOp(
          node, org_stack, org_stack_index_map, in_tensors, m_dmeta);
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
  std::unordered_map<CValPtr, IValPtrShared> m_value_ivalue_map;
};

void HandleDynamicOps(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dmeta) {
  PT_EAGER_TRACE;
  HandleDynamicOpsPass pass{graph, dmeta};

  bool changed{pass.run(stack)};
  if (changed) {
    PT_EAGER_DEBUG(__PRETTY_FUNCTION__, ": \n", *graph);
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
  for (auto dtensor_info : dmeta->ds_input_patching_list) {
    auto dtensor_indexes = dtensor_info.second;
    std::vector<torch::jit::IValue*> dtensor_list;
    std::vector<habana::graph::SymIntData> scalar_list;
    for (auto it = dtensor_indexes.begin(); it != dtensor_indexes.end(); it++) {
      stack.push_back(dmeta->ds_stack[*it]);
      dtensor_list.push_back(&(dmeta->ds_stack[*it]));
      scalar_list.push_back(dmeta->ds_tensor_to_scalar_map[*it]);
    }

    if (!is_first_launch) {
      dtensor_info.first(dtensor_list, scalar_list, stack);
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
