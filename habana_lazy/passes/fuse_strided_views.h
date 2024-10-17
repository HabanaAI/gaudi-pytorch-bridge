/******************************************************************************
 * Copyright (C) 2024 HabanaLabs, Ltd.
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
void fuse_strided_views(std::shared_ptr<torch::jit::Graph>& graph);
}; // namespace habana_lazy
