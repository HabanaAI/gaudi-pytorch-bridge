/*******************************************************************************
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
#include <vector>
#include "../components/program.h"

namespace habana {
namespace program {
namespace utils {

/*
 * Returns BFS based topological order of given graph
 */
std::vector<const torch::jit::Node*> BfsTopSort(const torch::jit::Graph& graph);

} // namespace utils
} // namespace program
} // namespace habana
