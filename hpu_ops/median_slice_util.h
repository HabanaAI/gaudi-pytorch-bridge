/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
    int nelements,
    int ndimension,
    int reduction_axis,
    int median_variant,
    bool final_node,
    c10::optional<int> node_index = c10::nullopt);

} // namespace habana
