/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace habana_lazy {
using Graph = torch::jit::Graph;
bool FoldConvBatchnorm(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack,
    std::vector<torch::jit::Value*>& redundant_inputs);
}; // namespace habana_lazy
