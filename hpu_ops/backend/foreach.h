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

#include "backend/helpers/habana_types.h"

namespace habana {

size_t computeInputsNumber(const at::Stack& stack);

typedef std::function<synapse_helpers::tensor(
    OpBackend*,
    synapse_helpers::graph&,
    std::string&,
    const std::vector<synTensor>&,
    const std::vector<at::IValue>&,
    int out_index)>
    NodeCreateFunction;
typedef std::function<SharedMetaDataVector(
    const at::Stack&,
    habana_helpers::HabanaExecutionMode executionMode)>
    SharedMetaCreateFunction;

std::vector<synapse_helpers::tensor> CommonForeachBinary(
    OpBackend* op,
    std::string& guid_,
    const std::vector<synTensor>& inputs,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    NodeCreateFunction node_creator);

SharedMetaDataVector CommonForeachBinarySharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode executionMode,
    SharedMetaCreateFunction sharedMetaCreator);
} // namespace habana