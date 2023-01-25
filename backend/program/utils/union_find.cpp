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

#include "union_find.h"
#include "../components/program.h"

namespace habana {
namespace program {
namespace utils {

namespace {

// Small utility to make behaviour of algorithm deterministic
void ReorderInts(std::int64_t& lhs, std::int64_t& rhs) {
  if (rhs > lhs) {
    std::swap(lhs, rhs);
  }
}

} // namespace

UnionFind::UnionFind(std::size_t n) {
  parent_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    parent_[i] = i;
  }
}

std::int64_t UnionFind::Find(std::int64_t x) {
  TORCH_CHECK_GE(x, 0);

  if (parent_.at(x) == x) {
    return x;
  }
  parent_.at(x) = Find(parent_.at(x));
  return parent_.at(x);
}

void UnionFind::Merge(std::int64_t lhs, std::int64_t rhs) {
  lhs = Find(lhs);
  rhs = Find(rhs);
  if (lhs == rhs)
    return;
  ReorderInts(lhs, rhs);
  parent_.at(lhs) = rhs;
}

bool UnionFind::Eq(std::int64_t lhs, std::int64_t rhs) {
  return Find(lhs) == Find(rhs);
}

void UnionFind::Substitute(SplittingDecision& decision) {
  for (auto& p : decision.colors) {
    p.second = Find(p.second);
  }
}

} // namespace utils
} // namespace program
} // namespace habana
