/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/passes/replace_inplace_ops_ds.h"

namespace habana {
namespace graph {
namespace pass {
#define PT_MAX_SHAPETENSOR_INPUT 10

struct HandleDynamicOpsPass {
  explicit HandleDynamicOpsPass(
      std::shared_ptr<torch::jit::Graph> graph,
      std::shared_ptr<DynamicGraphMetaData> dmeta,
      std::map<int64_t, std::vector<int64_t>>* input_new_base_sizes,
      std::vector<habana_helpers::RangeInfo>* range_infos)
      : m_graph(std::move(graph)), m_dmeta(std::move(dmeta)) {
    m_input_new_base_sizes = input_new_base_sizes;
    m_range_infos = range_infos;
  }

  bool run(torch::jit::Stack& stack) {
    propagateShape(stack);
    bool changed{processBlocks(m_graph->block(), stack)};
    return changed;
  }

 private:
  void createGraphInputStackIndexMap(GraphInputIndexMap& org_stack_index_map) {
    for (size_t idx = 0; idx < m_graph->inputs().size(); ++idx) {
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
      for (size_t i = 0; i < tensors.size(); ++i) {
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
    std::unordered_map<CValPtr, torch::jit::IValue> value_ivalue_map;
    // Before running SIF, keeping track of all tensors that were
    // zero-dimensional since during HybridSIF run for some ops,
    // InferOutputMeta is invoked, where all
    // zero-dimensional tensors are made one-dimensional
    std::vector<int> zero_dim_tensor_inds;
    zero_dim_tensor_inds.reserve(org_stack.size());
    int index = 0;
    for (auto& inp : org_stack) {
      if (inp.isTensor()) {
        if (inp.toTensor().dim() == 0)
          zero_dim_tensor_inds.push_back(index);
      }
      index++;
    }
    // Run SIF
    HabanaLaunchOpPT::RunHybridSif(m_graph, org_stack, value_ivalue_map);
    // Post SIF run, Reverting all changed tensors
    // back to zero-dimensional shape
    for (auto index : zero_dim_tensor_inds) {
      auto tensor_ = org_stack.at(index).toTensor();
      if (tensor_.dim() == 1)
        SET_SIZE_STRIDE_0D(tensor_);
    }

    for (auto val_ivalue : value_ivalue_map) {
      m_value_ivalue_map[val_ivalue.first] =
          std::make_shared<IVal>(val_ivalue.second);
    }
    // dump shapes
    // dumpValueIValueMap();
  }

  bool maxTensorDimsCheck(
      torch::jit::Node* node,
      const std::string& node_name) {
    // Checks that all input tensor sizes in the node
    // do not exceed "SYN_MAX_TENSOR_DIM" dimensions.
    int max_dim = SYN_MAX_TENSOR_DIM;

    // For some strided view ops
    // the tensor sizes should not exceed "SYN_MAX_TENSOR_DIM-1"
    // as their synapse implementation can cause tensor dim
    // expansion by under certain conditons
    static const std::set<std::string> strided_view_ops{
        "aten::as_strided", "aten::slice_scatter"};

    if (strided_view_ops.find(node_name) != strided_view_ops.end()) {
      if (node_name == "aten::as_strided") {
        auto ivalue =
            m_value_ivalue_map[const_cast<torch::jit::Value*>(node->input(2))];
        if (ivalue->isIntList()) {
          const auto strides = ivalue->toIntList();
          if (!strides.empty()) {
            int fcd_stride = strides.get(strides.size() - 1);
            // Dim expansion happens if
            // Fastest Changing Dimension is strided
            if (fcd_stride > 1)
              max_dim -= 1;
          }
        }
      } else if (node_name == "aten::slice_scatter") {
        auto ivalue =
            m_value_ivalue_map[const_cast<torch::jit::Value*>(node->input(1))];
        if (ivalue->isTensor()) {
          auto sizes = ivalue->toTensor().sizes();
          // Dim expansion does not happen
          // if any dim size value of src input tensor is 1
          auto it = std::find(sizes.begin(), sizes.end(), 1);
          if (it == sizes.end())
            max_dim -= 1;
        }
      }
    }

    for (const auto& input : node->inputs()) {
      auto ivalue = m_value_ivalue_map[const_cast<torch::jit::Value*>(input)];
      if (ivalue->isTensor()) {
        auto size = ivalue->toTensor().dim();
        if (size > max_dim) {
          PT_DYNAMIC_SHAPE_DEBUG(
              "Tensor size ",
              size,
              " in node ",
              node_name,
              " exceeds maximum dimensions support");
          return false;
        }
      }
    }
    return true;
  }

  bool nodeHasScalarGraphInput(
      torch::jit::Node* node,
      GraphInputIndexMap& org_stack_index_map) {
    for (const auto& input : node->inputs()) {
      torch::jit::Node* producer_node = input->node();
      if (producer_node->kind() == torch::jit::prim::ListConstruct)
        return nodeHasScalarGraphInput(producer_node, org_stack_index_map);
      else {
        auto ivalue = m_value_ivalue_map[const_cast<torch::jit::Value*>(input)];
        if (!ivalue->isTensor()) {
          if (org_stack_index_map.count(input->debugName())) {
            auto node_name = node->kind().toQualString();
            PT_EAGER_DEBUG(
                "Node ",
                node_name,
                " has scalar inputs that are also graph inputs");
            return true;
          }
        }
      }
    }
    return false;
  }

  bool isNodeDynamic(
      torch::jit::Node* node,
      GraphInputIndexMap& org_stack_index_map) {
    // Assuming node is dynamic by default
    bool isDynamic = true;
    auto node_name = node->kind().toQualString();
    auto outputshapes_attr = c10::Symbol::attr("output_shapes");
    if (node->hasAttribute(outputshapes_attr)) {
      auto outputshapes_str = node->s(outputshapes_attr);
      if (outputshapes_str.empty()) {
        PT_EAGER_DEBUG(
            "output_shapes attr is empty for node = ",
            node_name,
            ", assuming it to be dynamic");
      } else {
        bool hasSymbol = false;
        for (auto& c : outputshapes_str) {
          if (!(std::isdigit(c) || c == '[' || c == ']' || c == ',' ||
                std::isspace(c)))
            hasSymbol = true;
        }
        // Node is not dynamic if it does not have
        // any non-numeric symbols
        if (!hasSymbol)
          isDynamic = false;
        // If node has scalar inputs that are also graph inputs
        // Differing values of those inputs cause JIT cache miss
        // Better to replace such nodes
        if (nodeHasScalarGraphInput(node, org_stack_index_map))
          isDynamic = true;
      }
    } else {
      PT_EAGER_DEBUG(
          "output_shapes attr is missing for node = ",
          node_name,
          ", assuming it to be dynamic");
    }
    return isDynamic;
  }

  bool processBlock(torch::jit::Block* block, torch::jit::Stack& org_stack) {
    GraphInputIndexMap org_stack_index_map;
    createGraphInputStackIndexMap(org_stack_index_map);
    HABANA_ASSERT(m_graph->inputs().size() == org_stack.size());
    bool changed{true};
    // First Pass: Repace all dynamic shape ops with hpu implementation.
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      std::string node_name = it->kind().toQualString();
      torch::jit::Node* node{*it};
      if (!maxTensorDimsCheck(node, node_name))
        m_dmeta->static_fallback = true;
      DynamicOpPtr dsOp = DSOpsRegistry().get(node_name);
      if (!dsOp)
        continue;

      PT_EAGER_DEBUG("Replace dynamic Op: ", node_name);
      dsOp->m_input_new_base_sizes = m_input_new_base_sizes;
      dsOp->m_range_infos = m_range_infos;
      changed = dsOp->ReplaceWithDynamicHPUOp(
          node, org_stack, org_stack_index_map, m_value_ivalue_map, m_dmeta);
      if (!changed)
        PT_EAGER_DEBUG(
            "Replace with Dynamic HPU op failed for node ", node_name);
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
    bool changed{true};
    m_dmeta->static_fallback = false;
    for (auto block : blocks)
      changed &= processBlock(block, org_stack);
    return changed;
  }

  std::shared_ptr<torch::jit::Graph> m_graph;
  std::shared_ptr<DynamicGraphMetaData> m_dmeta;
  CValuePtrToIValuePtrMap m_value_ivalue_map;
  std::map<int64_t, std::vector<int64_t>>* m_input_new_base_sizes;
  std::vector<habana_helpers::RangeInfo>* m_range_infos;
};

void HandleDynamicOps(
    std::shared_ptr<torch::jit::Graph> graph,
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    std::map<int64_t, std::vector<int64_t>>* input_new_base_sizes,
    std::vector<habana_helpers::RangeInfo>* range_infos) {
  PT_EAGER_TRACE;
  // Replace inplace ops with out-of-place variant for which DS support is needed
  // Currently supports strided_insert_
  // This leverages DS support of out-of-place variant op for inplace variant
  ReplaceInplaceOpsDS(graph, habana::graph::DSOpsRegistry().getRegisteredDSOpsList());
  HandleDynamicOpsPass pass{graph, dmeta, input_new_base_sizes, range_infos};
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
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    LaunchDynamicShapes& launch_shapes) {
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
    dsOp->ResolveNegativeSizes(node, m_value_ivalue_map, launch_shapes);
  }
}

void HandleDynamicInputPatching(
    torch::jit::Stack& stack,
    std::shared_ptr<DynamicGraphMetaData> dmeta,
    LaunchDynamicShapes& launch_shapes,
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
  c10::SmallVector<
      std::vector<std::pair<int64_t, int64_t>>,
      PT_MAX_SHAPETENSOR_INPUT>
      mixed_list;
  for (auto dtensor_info : dmeta->ds_input_patching_list) {
    auto dtensor_indexes = dtensor_info.second;
    dtensor_list.clear();
    scalar_list.clear();
    tensor_list.clear();
    mixed_list.clear();
    for (auto it : dtensor_indexes) {
      stack.emplace_back(dmeta->ds_stack[it]);
      dtensor_list.emplace_back(&(dmeta->ds_stack[it]));
      scalar_list.emplace_back(dmeta->ds_tensor_to_scalar_map[it]);
      tensor_list.emplace_back(dmeta->ds_tensor_to_tensor_map[it]);
      mixed_list.emplace_back(dmeta->ds_mixed_map[it]);
    }

    if (!is_first_launch) {
      dtensor_info.first(
          dtensor_list,
          scalar_list,
          tensor_list,
          mixed_list,
          stack,
          launch_shapes);
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
