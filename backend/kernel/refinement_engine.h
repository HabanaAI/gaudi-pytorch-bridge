/******************************************************************************
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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "absl/types/optional.h"

namespace habana {

class RefinementEngine {
 private:
  std::atomic_bool m_refineFlag;
  std::mutex m_mutex;
  std::deque<absl::optional<size_t>> m_readyQueue;
  std::condition_variable m_refineCV;
  std::vector<std::thread> m_threads;

 public:
  static RefinementEngine& GetEngine();

  void Initialize();
  void Refine();
  void Shutdown();
  void AddGraphKey(size_t key);
};

} // namespace habana
