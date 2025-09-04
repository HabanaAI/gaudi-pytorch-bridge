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

#include "jit_fork/ir/ir.h"

namespace habana_torch::jit {

TORCH_API std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names = true);

TORCH_API void CanonicalizeOutputs(std::shared_ptr<Graph>& graph);

TORCH_API std::optional<const Use> firstOrLastUse(Value* v, bool find_first);

TORCH_API bool isBeforeOrAfter(
    const Use& a,
    const Use& b,
    bool checking_before);

} // namespace habana_torch::jit
