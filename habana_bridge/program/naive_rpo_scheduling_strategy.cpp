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

#include "naive_rpo_scheduling_strategy.h"

namespace habana {
namespace program {

SchedulingDecision NaiveRpoSchedulingStrategy(const GraphOfClusters& graph) {
  (void)graph;
  SchedulingDecision decision;
  // Not yet implemented
  return decision;
}

} // namespace program
} // namespace habana