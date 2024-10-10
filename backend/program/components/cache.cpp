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

#include "cache.h"

namespace habana {
namespace program {

ClusteredProgramSPtr Cache::Lookup(std::size_t graphKey) {
  std::lock_guard<std::mutex> _lck(mutex_);
  auto it = table_.find(graphKey);
  if (it == table_.end())
    return nullptr;
  return it->second;
}

void Cache::Insert(std::size_t graphKey, const ClusteredProgramSPtr& program) {
  std::lock_guard<std::mutex> _lck(mutex_);
  table_[graphKey] = program;
}

Cache& Cache::GetInstance() {
  static Cache instance;
  return instance;
}

} // namespace program
} // namespace habana