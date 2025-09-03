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

#include <backend/helpers/tensor_utils.h>
#include <memory>
#include <unordered_map>
#include "jit_fork/ir/ir.h"

namespace habana::sif_utils {
void mapGraphInputsToInputsOnStack(
    const std::shared_ptr<habana_torch::jit::Graph>& graph,
    const torch::jit::Stack&,
    std::unordered_map<CValPtr, habana_torch::jit::IValue>&);

c10::ScalarType getNodeScalarTypeFromInputs(
    const habana_torch::jit::Node*,
    const std::unordered_map<CValPtr, habana_torch::jit::IValue>&);

torch::jit::Stack createInputStackForNode(
    const habana_torch::jit::Node*,
    const std::unordered_map<CValPtr, habana_torch::jit::IValue>&);
} // namespace habana::sif_utils
