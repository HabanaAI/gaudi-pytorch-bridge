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

#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/ir/ir.h>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/passes/weight_permute_graph.h"
#include "habana_lazy/tensor_impl.h"

namespace habana_lazy {
using Graph = torch::jit::Graph;
void RecalculateBatchnormParams(
    std::shared_ptr<Graph>& graph,
    torch::jit::Stack& stack);
}; // namespace habana_lazy
