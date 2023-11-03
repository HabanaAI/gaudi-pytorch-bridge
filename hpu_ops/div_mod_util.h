/*******************************************************************************
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
enum DIV_MODE_OUTPUT_TYPE { QUOTIENT, REMAINDER, DIV_MODE_OUTPUT_TYPE_COUNT };
std::shared_ptr<void> FillDivModParams(size_t& size, bool pyCompatible = true);
std::vector<synapse_helpers::tensor> GetDivModOutput(
    OpBackend* pOpBackend,
    synapse_helpers::graph& graph,
    synTensor syn_numerator,
    synTensor syn_denominator,
    bool pyCompatible,
    const std::vector<long int> shape_out,
    DIV_MODE_OUTPUT_TYPE t);
} // namespace habana
