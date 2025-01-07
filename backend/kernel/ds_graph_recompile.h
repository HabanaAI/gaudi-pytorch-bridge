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

#include <algorithm>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include "backend/kernel/hpu_habana_launch_op_pt.h"

namespace habana {

at::Tensor CreateEmptyTensor(
    const PtTensorInfo& ti,
    habana::ShapeTensorStruct& tensor_data,
    const std::vector<int64_t>& tshape);

torch::jit::Stack CreateInputStack(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    std::unordered_map<uint64_t, habana::ShapeTensorStruct>& input_metadata,
    habana_helpers::TensorShapes& input_shapes);
void PrintStack(torch::jit::Stack& st);

bool CompileGraphWithRange(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    std::unordered_map<uint64_t, habana::ShapeTensorStruct>& input_metadata,
    habana_helpers::ResultShapes& input_ranges,
    habana_helpers::Bucket& new_bucket,
    size_t& new_recipe_key,
    std::shared_ptr<habana_helpers::CompilationStatistics> statpsh,
    std::shared_ptr<habana_helpers::DynamicBucketInfo> dbipsh);

bool RefineBucketDS(size_t graph_key);

} // namespace habana
