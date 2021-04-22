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
void replace_inplace_ops(std::shared_ptr<torch::jit::Graph>& graph);
}; // namespace habana_lazy
