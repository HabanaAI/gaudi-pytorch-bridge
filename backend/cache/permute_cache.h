/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include "backend/jit_graph_cache.h"

namespace habana {

/**
 * @brief Class to cache tensor output permutations for the jit graphs
 *
 * PermuteCache class stores PermutationWithOutputPosition vectors for
 * OptimizedJitGraphAndMetaData jit graphs in a static unordered map.
 * It is expected to be read from and written to from the LaunchThread
 *
 */
class PermuteCache {
 public:
  static std::optional<
      std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>>
  GetCachedPermute(const OptimizedJITGraphAndMetaData& jit_graph);
  static std::optional<
      std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>>
  GetCachedPermute(size_t shapeless_hash);
  static void CachePermuteForGraph(
      const OptimizedJITGraphAndMetaData& jit_graph,
      std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>
          permute_info);
  static void CachePermuteForGraph(
      size_t shapeless_hash,
      std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>
          permute_info);
  static std::string PermuteCacheDebugInfo();
  static size_t Size();
  static bool Empty();
  static void Flush();

 private:
  static inline std::shared_mutex smtx_{};
  static inline std::unordered_map<
      size_t,
      std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>>
      permute_cache_{};
};
} // namespace habana
