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
#include "../components/program.h"
#include "../components/strategy.h"

namespace habana {
namespace program {
namespace utils {

/*
 * UnionFind algorithm (a.k.a. Disjoint-set data structure).
 *
 * Used to merge colored partitions in FixColors/FixCycles.
 *
 * Implementation contains path-compression enhancement.
 *
 * See:
 *  https://en.wikipedia.org/wiki/Disjoint-set_data_structure
 */
class UnionFind {
 public:
  UnionFind(std::size_t n);

  // Find canonical representative of given element `x`
  std::int64_t Find(std::int64_t x);

  // Merge colors `lhs` and `rhs`
  void Merge(std::int64_t lhs, std::int64_t rhs);

  // Check if two given colors `lhs` and `rhs` are the same or already merged
  bool Eq(std::int64_t lhs, std::int64_t rhs);

  // Apply all merges into splitting decision
  void Substitute(SplittingDecision& decision);

 private:
  std::vector<std::int64_t> parent_;
};

} // namespace utils
} // namespace program
} // namespace habana
