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

#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>

#include "jit_fork/ir/alias_analysis.h"
#include "jit_fork/ir/ir.h"

#include <utility>

namespace habana_torch {
namespace jit {

struct TORCH_API MutationRemover {
  MutationRemover(
      std::shared_ptr<Graph> graph,
      std::optional<std::function<bool(Node*)>> mutation_filter = std::nullopt)
      : mutation_filter_(std::move(mutation_filter)),
        aliasDb_(nullptr),
        graph_(std::move(graph)) {}

  // return true if graph is modified
  bool removeListMutation();

  // return true if graph is modified
  bool removeTensorMutation();

  bool isSpecialMappedOp(Node* n) {
    return n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)") ||
        n->matches(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)") ||
        n->matches(
            "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)");
  }

  bool inplaceOpVariant(Node* n);

  static bool hasSideEffectOrAlias(Value* v, AliasDb* aliasDb);

 private:
  Node* createSpecialMappedOp(Node* n);
  bool listMutationFollowingListConstruct(Node* n);
  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  bool tryMakeUnaliasedIfOutputAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  // return true if graph is modified
  bool RemoveListMutation(Block* block);
  // return true if graph is modified
  bool RemoveTensorMutation(Block* block);

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  std::optional<std::function<bool(Node*)>> mutation_filter_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

// Removes list mutation with functional equivalents
// return true if graph is modified
TORCH_API bool RemoveListMutation(const std::shared_ptr<Graph>& graph);

// Replaces in-place aten ops with their functional equivalents
// when it can be proven that this does not change graph semantics
// if `mutation_filter` is present, the pass will only attempt to
// remove mutation on nodes which return true for the filter
// return true if graph is modified
TORCH_API bool RemoveTensorMutation(
    const std::shared_ptr<Graph>& graph,
    std::optional<std::function<bool(Node*)>> mutation_filter = std::nullopt);

// Replaces in-place aten activation ops with their functional equivalence
TORCH_API bool InplaceToFunctionalActivation(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace habana_torch
