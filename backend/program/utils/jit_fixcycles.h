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
#include "../components/strategy.h"
#include "union_find.h"

namespace habana {
namespace program {
namespace utils {

/*
 * Utility to remove cycles from colored graph.
 *
 * Merging colors (like in `FixColors`) can introduce cycles between colors what
 * would lead to invalid partitioning. This utility detects cycles and merges
 * colors on cycles.
 *
 * For example, if we have a cycle color1 -> color2 -> color3 -> color1, then
 * all those colors will be merged (replaced by one of them) to collapse cycling
 * partitions into one bigger.
 *
 * Utility will analyse given `decision` and `graph` and update
 * coloring-remapping represented by `uf`.
 */
void FixCycles(
    SplittingDecision& decision,
    UnionFind& uf,
    const torch::jit::Graph& graph);

} // namespace utils
} // namespace program
} // namespace habana
