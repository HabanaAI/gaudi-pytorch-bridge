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

c10::optional<synapse_helpers::tensor> HandleReductionDtype(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    synTensor syn_in,
    c10::optional<at::ScalarType> dtype);

std::shared_ptr<void> ReductionOpParams(
    const int ndim,
    size_t& size,
    int64_t index);

std::vector<synapse_helpers::tensor> HandleReductionDimAndKeepdim(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Tensor& self,
    std::vector<synTensor> inputs,
    const at::IntArrayRef dims,
    bool keepdim,
    const std::string& guid,
    std::vector<NodeAttr::NodeOutputAttr> output_attr);
} // namespace habana
