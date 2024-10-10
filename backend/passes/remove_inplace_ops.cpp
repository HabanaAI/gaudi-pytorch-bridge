#include "remove_inplace_ops.h"
#include <ATen/core/interned_strings.h>
using namespace torch::jit;

namespace habana {

static const std::unordered_map<std::string, std::string> inPlaceToOutOfPlace =
    {{"aten::add_", "aten::add"},
     {"hpu::add_", "hpu::add"},
     {"aten::div_", "aten::div"},
     {"aten::index_put_", "aten::index_put"},
     {"aten::mul_", "aten::mul"},
     {"aten::relu_", "aten::relu"},
     {"aten::clamp_", "aten::clamp"},
     {"aten::sub_", "aten::sub"}};

bool isInplaceOp(const Node* node) {
  return inPlaceToOutOfPlace.count(node->kind().toQualString()) != 0;
}

// Remove all in-place ops and replace them with out-of-place equivalents.
// e.g.
//   %foo = aten::add_(%foo, %n)
// becomes
//   %foo.2 = aten::add(%foo, %n)
//
// NOTE: this is NOT SAFE, since it assumes that the LHS is not aliased by
// another value. This is only to avoid breaking ONNX export; when alias
// analysis is done we can emit a warning if someone tries to export.
void RemoveInplaceOps(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      RemoveInplaceOps(block);
    }

    if (isInplaceOp(node)) {
      // create a replacement out of place op
      const std::string& newNodeStr =
          inPlaceToOutOfPlace.at(node->kind().toQualString());
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

void RemoveInplaceOps(const std::shared_ptr<Graph>& graph) {
  RemoveInplaceOps(graph->block());
}
} // namespace habana
