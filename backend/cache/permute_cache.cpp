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

#include "permute_cache.h"
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include "backend/jit_graph_cache.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

std::string PermuteCache::PermuteCacheDebugInfo() {
  std::ostringstream ss;
  ss << "PermuteCache size:" << PermuteCache::permute_cache_.size()
     << std::endl;
  return ss.str();
}

std::optional<
    std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>>
PermuteCache::GetCachedPermute(const size_t shapeless_hash) {
  std::shared_lock slck{PermuteCache::smtx_};

  auto itr = PermuteCache::permute_cache_.find(shapeless_hash);
  if (itr == PermuteCache::permute_cache_.end()) {
    PT_CACHE_DEBUG(
        "CACHE MISS:\t no saved permutations found for OptimizedJitGraphAndMetadata graph with shapeless_hash: ",
        shapeless_hash,
        "\n",
        PermuteCache::PermuteCacheDebugInfo());
    return std::nullopt;
  }

  PT_CACHE_DEBUG(
      "CACHE HIT:\t saved permutations found for OptimizedJitGraphAndMetadata graph with shapeless_hash: ",
      shapeless_hash,
      "\n",
      PermuteCache::PermuteCacheDebugInfo());
  return itr->second;
}

std::optional<
    std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>>
PermuteCache::GetCachedPermute(
    const OptimizedJITGraphAndMetaData& optimized_jit_graph) {
  return PermuteCache::GetCachedPermute(
      optimized_jit_graph.get_shapeless_with_dims_hash());
}

void PermuteCache::CachePermuteForGraph(
    const OptimizedJITGraphAndMetaData& optimized_jit_graph,
    std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>
        permutations) {
  PermuteCache::CachePermuteForGraph(
      optimized_jit_graph.get_shapeless_with_dims_hash(), permutations);
}

void PermuteCache::CachePermuteForGraph(
    size_t shapeless_hash,
    std::vector<OptimizedJITGraphAndMetaData::PermutationWithOutputPosition>
        permutations) {
  std::unique_lock lck{PermuteCache::smtx_};

  PermuteCache::permute_cache_.try_emplace(shapeless_hash, permutations);
  PT_CACHE_DEBUG(
      "Saving permutations for OptimizedJitGraphAndMetadata graph with shapeless_hash: ",
      shapeless_hash,
      "\n",
      PermuteCache::PermuteCacheDebugInfo());
}

size_t PermuteCache::Size() {
  std::shared_lock slck{PermuteCache::smtx_};
  return PermuteCache::permute_cache_.size();
}

bool PermuteCache::Empty() {
  std::shared_lock slck{PermuteCache::smtx_};
  return PermuteCache::permute_cache_.empty();
}

void PermuteCache::Flush() {
  std::unique_lock lck{PermuteCache::smtx_};
  PermuteCache::permute_cache_.clear();
}
} // namespace habana
