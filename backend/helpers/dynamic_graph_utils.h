/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include <torch/torch.h>
#include <cfloat>
#include "backend/helpers/tensor_utils.h"

using GraphInputIndexMap = std::unordered_map<std::string, int64_t>;
using InputSymbolMap = std::unordered_map<std::string, std::shared_ptr<double>>;

namespace habana_helpers {

bool is_symbolic_expr(const std::string& expr_str);

bool is_output_shape_empty(const std::string& expr_str);

bool nodeHasScalarGraphInput(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    CValuePtrToIValuePtrMap& value_ivalue_map);

bool isNodeDynamic(
    torch::jit::Node* node,
    GraphInputIndexMap& org_stack_index_map,
    CValuePtrToIValuePtrMap& value_ivalue_map);

void createGraphInputStackIndexMap(
    const std::shared_ptr<torch::jit::Graph>& graph,
    GraphInputIndexMap& org_stack_index_map);

size_t CalculateSymbolValuesHash(InputSymbolMap& symbol_value_map);

} // namespace habana_helpers