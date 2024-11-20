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

#include "replace_inplace_ops_ds.h"
#include <ATen/core/interned_strings.h>
using namespace torch::jit;

namespace habana {

// All inplace ops have node names ending with _ by naming convention
// Like hpu::strided_insert_
bool isInplaceOpDS(const Node* node) {
  const std::string& origNodeStrInplace = node->kind().toQualString();
  if (origNodeStrInplace.back() == '_')
    return true;
  return false;
}

// Replace all in-place ops with out-of-place equivalent for which DS support is needed.
// Inplace op is replaced only if a corresponding out-of-place op is registered as a DS
// op in HPU.
void ReplaceInplaceOpsDS(Block* block, const std::vector<std::string> DSOpsRegistryInplace) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      ReplaceInplaceOpsDS(block, DSOpsRegistryInplace);
    }

    if (isInplaceOpDS(node)) {

      // create a replacement out of place op
      const std::string& origNodeStrInplace = node->kind().toQualString();
      const std::string& newNodeStr =
          origNodeStrInplace.substr(0, origNodeStrInplace.length() - 1);
      // Check if the new node is registered in HPU as a Dynamic op
      // If not registered, skip replacing this inplace op as there is
      // no corresponding out-of-place DS op
      auto dsOp = std::find(DSOpsRegistryInplace.begin(), DSOpsRegistryInplace.end(), newNodeStr);
      if (dsOp == DSOpsRegistryInplace.end())
        continue;

      auto newNode = graph->create(Symbol::fromQualString(newNodeStr));
      newNode->copyAttributes(*node);
      newNode->insertBefore(node);
      newNode->setScope(node->scope());
      // copy inputs
      for (auto input : node->inputs()) {
        newNode->addInput(input);
      }

      // Create a new output node and replace all uses of self with it
      newNode->output()->copyMetadata(node->output());
      node->replaceAllUsesWith(newNode);
      node->inputs()[0]->replaceAllUsesAfterNodeWith(
          newNode, newNode->output());
      node->destroy();
    }
  }
}

void ReplaceInplaceOpsDS(const std::shared_ptr<Graph>& graph, const std::vector<std::string> DSOpsRegistryInplace) {
  ReplaceInplaceOpsDS(graph->block(), DSOpsRegistryInplace);
}
} // namespace habana
