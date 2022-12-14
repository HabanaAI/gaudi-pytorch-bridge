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
#include "strategy.h"

namespace habana {
namespace program {

/*
 * Naive reverse post order (dfs-based topological order) scheduling strategy.
 *
 * Just computes topological order of given graph, does not take anything into
 * account.
 */
SchedulingDecision NaiveRpoSchedulingStrategy(const GraphOfClusters& graph);

} // namespace program
} // namespace habana