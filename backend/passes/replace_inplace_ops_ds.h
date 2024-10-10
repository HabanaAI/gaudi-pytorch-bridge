/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include <torch/csrc/jit/ir/ir.h>

namespace habana {
void ReplaceInplaceOpsDS(const std::shared_ptr<torch::jit::Graph>& graph, const std::vector<std::string> DSOpsRegistryInplace);
} // namespace habana
