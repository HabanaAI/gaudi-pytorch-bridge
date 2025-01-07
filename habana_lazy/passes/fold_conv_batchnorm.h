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

#include <torch/csrc/jit/ir/ir.h>

namespace habana_lazy {
using Graph = torch::jit::Graph;
bool FoldConvBatchnorm(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    std::vector<torch::jit::Value*>& redundant_inputs);
}; // namespace habana_lazy
