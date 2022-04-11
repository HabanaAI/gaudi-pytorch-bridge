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
#include "hpu_op_helper.h"

namespace habana {

std::shared_ptr<void> ReductionOpParams(
    const int ndim,
    size_t& size,
    int64_t index);

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> inputs,
    at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    const at::IntArrayRef self_shape,
    const at::IntArrayRef outshape,
    std::vector<NodeAttr::NodeOutputAttr> output_attr);
} // namespace habana
