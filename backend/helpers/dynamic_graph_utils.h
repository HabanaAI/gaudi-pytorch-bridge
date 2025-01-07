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