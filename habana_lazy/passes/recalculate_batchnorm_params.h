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

#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/ir/ir.h>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/tensor_impl.h"

#include <tuple>

namespace habana_lazy {
using Graph = torch::jit::Graph;
using Node = torch::jit::Node;

::std::tuple<habana::TensorExtraMeta*, habana::StorageExtraMeta*>
GetBackEndTensorMeta(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx);

void* GetDataInHostBuffer(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx);

void UpdateDataInDeviceMem(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    Node* node,
    const int idx,
    void* host_ptr);

void RecalculateBatchnormParams(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack);
}; // namespace habana_lazy
