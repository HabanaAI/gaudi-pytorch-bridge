/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#pragma once

#include <backend/helpers/tensor_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>
#include <unordered_map>

namespace habana::sif_utils {
void mapGraphInputsToInputsOnStack(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const torch::jit::Stack&,
    std::unordered_map<CValPtr, torch::jit::IValue>&);

c10::ScalarType getNodeScalarTypeFromInputs(
    const torch::jit::Node*,
    const std::unordered_map<CValPtr, torch::jit::IValue>&);

torch::jit::Stack createInputStackForNode(
    const torch::jit::Node*,
    const std::unordered_map<CValPtr, torch::jit::IValue>&);
} // namespace habana::sif_utils
