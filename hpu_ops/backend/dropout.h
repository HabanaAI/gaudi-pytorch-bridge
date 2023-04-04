/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/op_backend.h"

namespace ns_DropoutKernel {
struct Params;
}

namespace habana {
std::vector<synapse_helpers::tensor> BuildDropout(
    OpBackend*,
    synapse_helpers::graph&,
    const std::vector<OpBackend::TensorsPair>&,
    const std::vector<NodeAttr::NodeOutputAttr>&,
    ns_DropoutKernel::Params*,
    const size_t);
}
