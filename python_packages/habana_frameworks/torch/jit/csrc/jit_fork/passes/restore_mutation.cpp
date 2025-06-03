/**
 * Copyright (c) 2021-2025 Intel Corporation
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
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>

#include "remove_mutation.h"
#include "restore_mutation.h"

namespace habana_torch {
namespace jit {

FunctionalToInplaceRewriter::FunctionalToInplaceRewriter(
    std::shared_ptr<Graph> graph)
    : aliasDb_(nullptr), graph_(std::move(graph)) {}

bool FunctionalToInplaceRewriter::CanBeInplace(Node* node) {
  if (activation_type_promotion_mapping.find(node->kind()) ==
      activation_type_promotion_mapping.end()) {
    return false;
  }

  Symbol inplace_op =
      Symbol::fromQualString(std::string(node->kind().toQualString()) + "_");
  if (!inplace_op) {
    return false;
  }

  // If type promotion is allowed, then perform dtype check
  bool check_dtype = activation_type_promotion_mapping.at(node->kind());

  Value* input = node->inputs().at(0);
  Value* output = node->outputs().at(0);
  auto inputDtype = input->type()->expect<TensorType>()->scalarType();
  auto outputDtype = output->type()->expect<TensorType>()->scalarType();

  // In general, we don't need to check shape for activation ops as they
  // element-wise. But for those where type promotion could happen, we need to
  // make sure the dtype of input and output are the same. For now the dtype
  // checking will always fail until the type inference is ready.
  if (check_dtype &&
      (!inputDtype || !outputDtype ||
       inputDtype.value() != outputDtype.value())) {
    return false;
  }

  // Skip if input's def node has side effect or input has alias
  if (MutationRemover::hasSideEffectOrAlias(input, getOrCreateAliasDb())) {
    return false;
  }

  // If x has more than one use, skip the conversion.
  // TODO: Use liveness analysis to catch more general scenario
  return (input->uses().size() == 1);
}

bool FunctionalToInplaceRewriter::FunctionalToInplace(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= FunctionalToInplace(sub_block);
    }

    if (!CanBeInplace(node)) {
      continue;
    }

    changed = true;
    Node* inplace_node = node->replaceWithNewSymbol(
        Symbol::fromQualString(node->schema().name() + "_"));
    inplace_node->output()->replaceAllUsesWith(node->inputs().at(0));
    getOrCreateAliasDb()->replaceWithNewValue(
        node->output(), inplace_node->output());

    node->destroy();
  }
  return changed;
}

bool FunctionalToInplaceActivation(const std::shared_ptr<Graph>& graph) {
  FunctionalToInplaceRewriter rewriter(graph);
  return rewriter.FunctionalToInplace(graph->block());
}

} // namespace jit
} // namespace habana_torch
