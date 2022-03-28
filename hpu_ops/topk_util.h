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

std::vector<synapse_helpers::tensor> TopK_Helper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    int reduction_axis,
    const at::IntArrayRef topk_outshape,
    int descending_order,
    int ndimension,
    int kvalue,
    int variant);

} // namespace habana
