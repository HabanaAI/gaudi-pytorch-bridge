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
#include <memory>
#include "hpu_op_helper.h"

namespace habana {

std::vector<synapse_helpers::tensor> Median_Slice_Helper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    const at::ScalarType dtype,
    int64_t nelements,
    int64_t ndimension,
    int64_t reduction_axis,
    int64_t median_variant,
    bool final_node,
    c10::optional<int> node_index = c10::nullopt);

} // namespace habana
