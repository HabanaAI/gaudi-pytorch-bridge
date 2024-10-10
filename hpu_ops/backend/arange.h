/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
