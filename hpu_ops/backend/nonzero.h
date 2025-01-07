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
#include <ATen/core/DimVector.h>
#include <c10/util/Optional.h>
#include "hpu_ops/nonzero.h"

namespace habana {
std::vector<synapse_helpers::tensor> NonZeroCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    NonZeroParams_t self_params,
    synTensor self_synin,
    c10::optional<int> final_result_index_0,
    c10::optional<int> final_result_index_1,
    bool use_tpc_impl = false);
} // namespace habana
