/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/hpu_op_helper.h"

namespace habana {
std::vector<int64_t> ComputeOutputShapeWithAdvIndexing(
    std::vector<int64_t> input_shape,
    at::TensorList indices,
    c10::List<int64_t> adv_index_dims,
    bool get_adv_indexing_out_shape);
} // namespace habana
