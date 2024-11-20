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
#include <memory>
#include "hpu_op_helper.h"

namespace habana {
enum DIV_MODE_OUTPUT_TYPE { QUOTIENT, REMAINDER, DIV_MODE_OUTPUT_TYPE_COUNT };
std::shared_ptr<void> FillDivModParams(size_t& size, bool pyCompatible = true);
std::vector<synapse_helpers::tensor> GetDivModOutput(
    OpBackend* pOpBackend,
    synapse_helpers::graph& graph,
    synTensor syn_numerator,
    synTensor syn_denominator,
    bool pyCompatible,
    const std::vector<long int> shape_out,
    DIV_MODE_OUTPUT_TYPE t);
} // namespace habana
