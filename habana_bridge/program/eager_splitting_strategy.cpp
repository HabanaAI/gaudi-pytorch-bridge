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

#include "eager_splitting_strategy.h"

namespace habana {
namespace program {

SplittingDecision EagerSplittingStrategy(const LazyJitGraph& lazy_graph) {
  SplittingDecision decision;
  std::int64_t fresh_color = 0;
  const auto& graph = *lazy_graph.get_cached_graph();
  for (auto* node : graph.nodes()) {
    decision.colors[node] = fresh_color++;
  }

  return decision;
}

} // namespace program
} // namespace habana