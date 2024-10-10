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

namespace habana {
namespace program {

/*
 * Splitting strategy based on bold heuristic using liveness analysis.
 * It splits the graph at places where number of bytes alive (according
 * to heuristics) spills above threshold.
 */
SplittingDecision LvaSpillSplittingStrategy(const LazyJitGraph& graph);

} // namespace program
} // namespace habana