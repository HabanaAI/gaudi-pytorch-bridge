/*******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include "program.h"
#include "strategy.h"

namespace habana {
namespace program {

/*
 * TODO
 *
 * Represents partitioned graph and its relations to clusters associated to
 * other colors by strategy.
 */
struct ClusterAfterSplitting {
  std::shared_ptr<LazyJitGraph> graph_;
};

/*
 * TODO
 *
 * Describes result of splitting. Partitions associated to colors and data
 * dependencies related to inputs/outputs of original graph.
 */
struct SplittingResult {
  std::unordered_map<std::int64_t, ClusterAfterSplitting> clusters;
};

SplittingResult SplitJitIrGraph(
    const LazyJitGraph& graph,
    const SplittingDecision& decision);

} // namespace program
} // namespace habana