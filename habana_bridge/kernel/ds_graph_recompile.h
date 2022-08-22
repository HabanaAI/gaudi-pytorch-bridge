/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include <algorithm>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"

namespace habana {

at::Tensor CreateEmptyTensor(
    const PtTensorInfo& ti,
    const std::vector<int64_t>& tshape);

torch::jit::Stack CreateInputStack(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    habana_helpers::TensorShapes& input_shapes);
void PrintStack(torch::jit::Stack& st);

bool CompileGraphWithRange(
    std::shared_ptr<habana::RecipeValueSpec> rvpsh,
    habana_helpers::ResultShapes& input_ranges,
    habana_helpers::Bucket& new_bucket,
    size_t& new_recipe_key,
    std::shared_ptr<habana_helpers::CompilationStatistics> statpsh,
    std::shared_ptr<habana_helpers::DynamicBucketInfo> dbipsh);

bool RefineBucketDS(size_t graph_key, size_t step);

} // namespace habana
