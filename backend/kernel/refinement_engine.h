/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
