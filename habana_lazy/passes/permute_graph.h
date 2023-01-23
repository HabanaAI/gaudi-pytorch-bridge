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
#include "backend/habana_operator.h"
namespace habana_lazy {
void InsertPermute_graph(
    std::shared_ptr<torch::jit::Graph>& graph,
    torch::jit::Stack& stack);
}; // namespace habana_lazy
