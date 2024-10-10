/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <ATen/core/ATen_fwd.h>
#include <cstdint>
#include <vector>

#pragma once

namespace habana {
std::vector<int64_t> ComputeIndexOperatorOutputShape(
    const at::Tensor& input,
    at::TensorList indices);

std::vector<int64_t> ComputeGatherOperatorOutputShape(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index);
} // namespace habana