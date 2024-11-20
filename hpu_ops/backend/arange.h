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

namespace habana {
synapse_helpers::tensor ArangeCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    c10::Scalar start,
    c10::Scalar end,
    c10::Scalar step,
    c10::ScalarType out_dtype,
    std::optional<synTensor> syn_in0,
    std::optional<synTensor> syn_in1,
    std::string guid,
    std::vector<int64_t> outshape,
    std::shared_ptr<void> params,
    size_t size,
    c10::optional<int> final_result_index,
    bool is_eager = false);
std::shared_ptr<void> FillArangeParamsInternal(
    c10::Scalar start,
    c10::Scalar end,
    c10::Scalar step,
    c10::ScalarType out_scalar_type,
    size_t& size);
} // namespace habana
