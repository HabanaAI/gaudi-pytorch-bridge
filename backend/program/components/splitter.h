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
#include "program.h"
#include "strategy.h"

namespace habana {
namespace program {

// special color assigned to inputs
constexpr std::int64_t COLOR_PARAM = -1;
// special color assigned to outputs
constexpr std::int64_t COLOR_RETURN = -2;
// special color assigned to prim::Constant which needs to be duplicated
constexpr std::int64_t COLOR_CONSTANT_FOR_DUPLICATION = -3;

/*
 * Represents cluster cut from graph during splitting.
 */
struct ClusterAfterSplitting {
  std::shared_ptr<LazyJitGraph> graph_;

  // Maps output index into pairs (color, input_index)
  std::vector<PortVector> outputs_;
};

/*
 * Describes result of splitting.
 */
struct SplittingResult {
  // Maps input index into pairs (color, input_index)
  std::vector<PortVector> inputs_;
  // Partitions
  std::unordered_map<std::int64_t, ClusterAfterSplitting> clusters_;
};

/*
 * Raw splitting algorithm.
 * TODO: better name
 */
SplittingResult SplitJitIrGraph(
    const LazyJitGraph& graph,
    const SplittingDecision& decision);

/*
 * Creates GoC from splitting algorithm result.
 * TODO: better name
 */
std::unique_ptr<GraphOfClusters> CreateGraphOfClustersFromSplittingResult(
    SplittingResult& result);

/*
 * TODO: better name
 */
std::unique_ptr<GraphOfClusters> SplitIntoGoc(
    const LazyJitGraph& graph,
    const SplittingDecision& decision);

} // namespace program
} // namespace habana