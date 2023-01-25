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
#include <functional>
#include <unordered_map>
#include <vector>
#include "program.h"
#include "torch/csrc/jit/ir/ir.h"

namespace habana {
namespace program {

/*
 * Result of spliting strategy is represented by class SplittingDecision.
 * The strategy is expected to map every graph node into a color represented by
 * non negative integer (negative colors are reserved for splitting algorithm).
 *
 * Colors of special nodes representing inputs/outputs are irrevelant.
 *
 * Optionaly, splitting strategy can propose scheduling of created partitions,
 * it should be represented by exeuction order of colors.
 * TODO: maybe it unnecessary and should be removed?
 *
 * The method `Validate` can be used to validate correctness of decision.
 *  - no introduced cycles
 *  - all nodes are mapped to nonnegative colors
 */
struct SplittingDecision {
  std::unordered_map<const torch::jit::Node*, std::int64_t> colors;

  explicit operator bool() const;
  bool Validate(const LazyJitGraph& lazy_graph) const;

  std::int64_t MaxColor() const;
};

/*
 * Result of scheduling strategy is represented by class SchedulingDecision.
 */
struct SchedulingDecision {
  std::vector<Cluster::Id> scheduling;
  explicit operator bool() const;
  bool Validate(const GraphOfClusters& graph) const;
};

/*
 * Splitting strategy.
 *
 * Should return SplittingDecision for given JIT IR Graph.
 * If returned decision is empty, then we fallback to eager-like strategy.
 */
using SplittingStrategy = std::function<SplittingDecision(const LazyJitGraph&)>;

/*
 * Schediling strategy.
 *
 * Should return SchedulingDecision for given Graph of Clusters.
 * If returned decision is empty, then we fallback to naive rpo strategy.
 */
using SchedulingStrategy =
    std::function<SchedulingDecision(const GraphOfClusters&)>;

/*
 * Obtains splitting strategy.
 *
 * Uses PT_HPU_CLUSTER_PROGRAM_SPLIT_STR environment variable to determine
 * splitting strategy.
 */
SplittingStrategy GetSplittingStrategy();

/*
 * Obtains scheduling strategy.

 * Uses PT_HPU_CLUSTER_PROGRAM_SCHED_STR environment variable to determine
 * splitting strategy.
 */
SchedulingStrategy GetSchedulingStrategy();

} // namespace program
} // namespace habana