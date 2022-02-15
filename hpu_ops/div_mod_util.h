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
enum DIV_MODE_OUTPUT_TYPE { QUOTIENT, REMAINDER, DIV_MODE_OUTPUT_TYPE_COUNT };
std::shared_ptr<void> FillDivModParams(size_t& size, bool pyCompatible = true);
std::vector<synapse_helpers::tensor> GetDivModOutput(
    OpBackend* pOpBackend,
    synapse_helpers::graph& graph,
    synTensor syn_numerator,
    synTensor syn_denominator,
    bool pyCompatible,
    const std::vector<long int> shape_out,
    const at::ScalarType result_type,
    DIV_MODE_OUTPUT_TYPE t);
} // namespace habana