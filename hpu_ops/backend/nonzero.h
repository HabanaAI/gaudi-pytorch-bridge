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
#include <ATen/core/DimVector.h>
#include <c10/util/Optional.h>
#include "hpu_ops/nonzero.h"

namespace habana {
std::vector<synapse_helpers::tensor> NonZeroCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    NonZeroParams_t self_params,
    synTensor self_synin,
    c10::optional<int> final_result_index_0,
    c10::optional<int> final_result_index_1,
    bool use_tpc_impl = false);
} // namespace habana
