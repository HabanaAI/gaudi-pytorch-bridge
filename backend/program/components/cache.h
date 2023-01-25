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
#include <mutex>
#include <unordered_map>
#include "program.h"

namespace habana {
namespace program {

class Cache {
 public:
  ClusteredProgramSPtr Lookup(std::size_t graphKey);
  void Insert(std::size_t graphKey, const ClusteredProgramSPtr& program);

  static Cache& GetInstance();

 private:
  Cache() = default;
  Cache(const Cache&) = delete;
  Cache(Cache&&) = delete;
  Cache& operator=(const Cache&) = delete;
  Cache& operator=(Cache&&) = delete;

  std::mutex mutex_;
  std::unordered_map<std::size_t, ClusteredProgramSPtr> table_;
};

} // namespace program
} // namespace habana
