/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "getitem_folding_pass.h"
#include <iostream>
#include <sstream>
#include <string>
#include "habana_helpers/logging.h"
namespace habana_torch {
namespace jit {
namespace {

class ProcessGetItemNodes {
 public:
  bool run(habana_torch::jit::Graph& g) {
    PT_BRIDGE_DEBUG("Starting 'GetItem folding' pass.");
    for (Node* n : g.nodes()) {
      // Collect all the list unpack nodes as we go
      collectUnpackNodes(n);
      // Have we found __getitem__ node?
      if (isGetItemNodeWithConst(n)) {
        const auto inputs = n->inputs();
        Value* dest_value = getDestValue(inputs);
        if (dest_value != nullptr) {
          Value* getitem_value = n->outputs()[GETITEM_ARG];
          PT_BRIDGE_DEBUG(
              "Replacing: ",
              getitem_value->debugName(),
              " with ",
              dest_value->debugName(),
              ".");
          getitem_value->replaceAllUsesWith(dest_value);
          graph_changed = true;
          // Collect debug names
          if (value_new_names_map.find(dest_value) == value_new_names_map.end())
            value_new_names_map[dest_value] = getitem_value->debugName();

          removal_getitem_nodes.insert(n);
          // This const node can potentially be removed
          // and it will have no other uses we will try to remove it
          Node* const_node = inputs[INDEX_ARG]->node();
          removal_const_nodes.insert(const_node);
        }
      }
    }
    cleanUp();
    return graph_changed;
  }

 private:
  const std::optional<int> getIndex(const Value* index_value) const {
    const auto index_ivalue = toIValue(index_value);
    // We assume the index value is int
    if (!index_ivalue.has_value() || !index_ivalue.value().isInt())
      return std::nullopt;
    // Convert from ivalue to int
    return index_ivalue.value().toInt();
  }

  Node* getUnpackNode(Value* container_value) const {
    const auto it = unpack_map.find(container_value);
    if (it == unpack_map.end())
      // Not found
      return nullptr;
    // Container was unpacked
    return it->second;
  }

  Value* getDestValue(const at::ArrayRef<Value*>& inputs) {
    // Obtain index value from getitem node (the second argument)
    const std::optional<int> index = getIndex(inputs[INDEX_ARG]);
    if (!index.has_value())
      return nullptr;

    Value* dest_value = nullptr;
    // Get container construction node such as:
    //  %getitem : Float(shape=[...], strides=[...], ..) | OUTPUT[0]
    //   = aten::__getitem__(                            | NODE
    //       %16,                                        | INPUT[0]
    //       %15                                         | INPUT[1]
    //)
    Node* container_node = inputs[CONTAINER_ARG]->node();
    // First argument input[0] is either list or tuple
    if (container_node->kind() == prim::ListConstruct ||
        container_node->kind() == prim::TupleConstruct) {
      // The list would like like
      // %16 : Tensor[]                                   | OUTPUT[0]
      //  = prim::ListConstruct(                          | NODE
      //       %10,                                       | INPUT[0]
      //       %11                                        | INPUT[1]
      //)
      // Here we obtain value from the container at given 'index'
      // and inputs are elements of the container.
      dest_value = container_node->inputs()[index.value()];
      removal_container_construct.insert(container_node);
    } else {
      // Find nodes such as:
      // %getitem : Long(shape=[...], strides=[...], ...) | OUTPUT[0]
      //  = aten::__getitem__(                            | NODE
      //      %split_with_sizes,                          | INPUT[0]
      //      %8                                          | INPUT[1]
      //  )

      // Check if container was unpacked at some point before
      Node* unpack_node = getUnpackNode(inputs[CONTAINER_ARG]);
      if (unpack_node == nullptr)
        return nullptr;
      // Here we obtain value from the container at given 'index'
      // and outputs of the unpack are elements of the continer
      dest_value = unpack_node->outputs()[index.value()];
    }
    return dest_value;
  }

  void collectUnpackNodes(Node* n) {
    if (n->kind() == prim::ListUnpack || n->kind() == prim::TupleConstruct) {
      // Find nodes such as:
      //   %6 : Long(shape=[...], strides=[...], ...), | OUTPUT[0]
      //   %7 : Long(shape=[...], strides=[...], ...)  | OUTPUT[1]
      //      = prim::ListUnpack(                      | NODE
      //            %split_with_sizes                  | INPUT[0]
      //      )
      // Input here will be argument to ListUnpack function ->
      // "%split_with_sizes"
      const auto inputs = n->inputs();
      unpack_map[inputs[CONTAINER_ARG]] = n;
    }
  }

  bool isGetItemNodeWithConst(const Node* n) const {
    // We are only interested in aten::__getitem__ and prim::TupleIndex nodes
    if (!(n->kind() == aten::__getitem__ || n->kind() == prim::TupleIndex))
      return false;
    // The node needs to have 2 arguments - container and index
    const auto inputs = n->inputs();
    if (inputs.size() != 2) {
      return false;
    }
    // The second argument (index) must be constant
    if (inputs[INDEX_ARG]->node()->kind() != prim::Constant) {
      return false;
    }
    PT_BRIDGE_DEBUG("Found __getitem__ node: ", *n, ".");
    return true;
  }

  bool isNumber(c10::string_view str) const {
    return str.find_first_not_of("0123456789") == std::string::npos;
  }

  static void destroyNodeIfHasNoUses(Node* n) {
    if (!n->hasUses()) {
      PT_BRIDGE_DEBUG("Removing node ", *n, ".");
      n->destroy();
    }
  };

  void cleanUp() {
    for_each(
        removal_getitem_nodes.begin(),
        removal_getitem_nodes.end(),
        ProcessGetItemNodes::destroyNodeIfHasNoUses);
    for_each(
        removal_const_nodes.begin(),
        removal_const_nodes.end(),
        ProcessGetItemNodes::destroyNodeIfHasNoUses);
    for_each(
        removal_container_construct.begin(),
        removal_container_construct.end(),
        ProcessGetItemNodes::destroyNodeIfHasNoUses);
    for_each(unpack_map.begin(), unpack_map.end(), [&](auto& value) {
      Node* container_node = value.second->inputs()[CONTAINER_ARG]->node();
      ProcessGetItemNodes::destroyNodeIfHasNoUses(value.second);
      ProcessGetItemNodes::destroyNodeIfHasNoUses(container_node);
    });
    // Rename the nodes
    for_each(
        value_new_names_map.begin(),
        value_new_names_map.end(),
        [&](auto& value) {
          if (!isNumber(value.second))
            PT_BRIDGE_DEBUG(
                "Renaming value ",
                value.first->debugName(),
                " to ",
                value.second,
                ".");
          value.first->setDebugName(value.second);
        });
  }

  static constexpr uint32_t GETITEM_ARG = 0;
  static constexpr uint32_t CONTAINER_ARG = 0;
  static constexpr uint32_t INDEX_ARG = 1;
  bool graph_changed = false;
  // Here we store all the nodes that will potentially be removed
  // Each type will be stored in different set because order of removal
  // is important. For example getitem node can use const and list construct
  // thus 'getitem' node must be removed in the first place.
  std::set<Node*> removal_getitem_nodes;
  std::set<Node*> removal_const_nodes;
  std::set<Node*> removal_container_construct;
  // This map will store all the nodes where unpack happens
  // The key is the first argument to unpack - the list on which unpack happens
  std::map<Value*, Node*> unpack_map;
  // Map to store new names for values
  std::map<Value*, std::string> value_new_names_map;
};

} // namespace

bool GetItemFoldingPass(habana_torch::jit::Graph& g) {
  return ProcessGetItemNodes().run(g);
}

} // namespace jit
} // namespace habana_torch
