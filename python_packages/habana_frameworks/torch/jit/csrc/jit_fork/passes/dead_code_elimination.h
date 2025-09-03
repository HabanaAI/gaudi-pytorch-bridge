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

#pragma once

#include "jit_fork/ir/ir.h"

namespace habana_torch::jit {

// If given a top-level graph, DCE will construct do alias analysis that allows
// for "smarter" dead code elimination (we will eliminate mutable ops if we can
// prove the mutated values are not used). Otherwise, we will not allow DCE to
// eliminate mutable ops.
//
// So, prefer to use the graph version if you can.
enum class DCESideEffectPolicy : uint8_t {
  // default behavior: dead code elimination will check if a node has side
  // effects
  // and not delete it if it does.
  DONT_DELETE_NODES_WITH_SIDE_EFFECTS,
  // with this flag, dead code elimination will not check if a node has side
  // effects and treat nodes with side effects like any other node,
  // i.e. delete them if their outputs aren't used anywhere.
  ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS
};

TORCH_API void EliminateDeadCode(
    const std::shared_ptr<Graph>& graph,
    DCESideEffectPolicy sideEffectPolicy =
        DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);
TORCH_API void EliminateDeadCode(
    Block* block,
    bool recurse = true,
    DCESideEffectPolicy sideEffectPolicy =
        DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);

// Invoke the user-provided callback on all live values before deleting anything
TORCH_API void EliminateDeadCode(
    Block* block,
    std::function<void(const std::unordered_set<const Value*>&)> cb,
    DCESideEffectPolicy sideEffectPolicy =
        DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);
} // namespace habana_torch::jit
