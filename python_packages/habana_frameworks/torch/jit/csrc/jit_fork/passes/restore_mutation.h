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

#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Export.h>

#include "jit_fork/ir/alias_analysis.h"
#include "jit_fork/ir/ir.h"

namespace habana_torch {
namespace jit {

// A map which stores if an activation operator can perform type promotion
const std::unordered_map<Symbol, bool> activation_type_promotion_mapping = {
    {aten::sigmoid, true},
    {aten::tanh, true},
    {aten::celu, false},
    {aten::elu, false},
    {aten::gelu, false},
    {aten::glu, false},
    {aten::hardshrink, false},
    {aten::hardsigmoid, false},
    {aten::hardswish, false},
    {aten::hardtanh, false},
    {aten::leaky_relu, false},
    {aten::prelu, false},
    {aten::relu6, false},
    {aten::relu, false},
    {aten::rrelu, false},
    {aten::selu, false},
    {aten::silu, false}};

class FunctionalToInplaceRewriter {
 public:
  FunctionalToInplaceRewriter(std::shared_ptr<Graph> graph);

  bool FunctionalToInplace(Block* block);

 private:
  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  bool CanBeInplace(Node* node);

  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

// A common application scenario is to apply InplaceToFunctionalActivation
// before some JIT optimization passes, so that those passes are less
// constrained by in-place ops. After those passes are done, we can call
// FunctionalToInplaceActivation to recover in-place activation ops,
// so that we won't lose the performance benefit coming from memory reduction.

// Replaces functional aten activation ops with their in-place equivalents
TORCH_API bool FunctionalToInplaceActivation(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace habana_torch
