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
 * Results of liveness analysis.
 */
struct LivenessResult {
  std::vector<std::size_t> bytes_alive_before;
  std::vector<std::size_t> bytes_alive_after;

  // If schedule is given, then output will be annotated by operations
  std::string ToDebugString(
      const std::vector<const torch::jit::Node*>& schedule = {}) const;
};

/*
 * Performs liveness analysis on given graph assuming given schedule
 */
LivenessResult AnalyseLiveness(
    const LazyJitGraph& lazy_graph,
    const std::vector<const torch::jit::Node*>& schedule);

} // namespace utils
} // namespace program
} // namespace habana
