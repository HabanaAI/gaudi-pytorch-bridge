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
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include "habana_helpers/habana_serialization/include/habana_serialization/recipe_cache.h"

#define PGM_LRU_MAX_EAGER_NRECIPES 100000
#define PGM_LRU_MAX_LAZY_NRECIPES 30000
#define PGM_LRU_MIN_NRECIPES 3

namespace habana {

class RecipeHolder;
class RecipeArgumentSpec;

struct RecipeArgumentSpecHash {
 public:
  size_t operator()(const std::shared_ptr<RecipeArgumentSpec>& v) const;
};

// Comparator for RecipeArgumentSpec
struct RecipeArgumentSpecEqual {
 public:
  bool operator()(
      const std::shared_ptr<RecipeArgumentSpec>& v1,
      const std::shared_ptr<RecipeArgumentSpec>& v2) const;
};

class DiskCache {
 public:
  DiskCache(const serialization::RecipeCacheConfig& recipe_cache_config);
  void Add(const RecipeHolder& val, const RecipeArgumentSpec& spec);
  std::shared_ptr<RecipeHolder> Find(const RecipeArgumentSpec& spec);
  // in case RecipeValueSpec creation failed, DiskCache is leaving lock files on
  // disk. This ensures a cleanup.
  void flush();

 private:
  serialization::RecipeCache recipe_cache_;
  // library specific suffix to determine for what TF, Synapse, etc. the cache
  // entry was produced
  std::string cache_id_suffix_;
};

class RecipeCacheLRU {
 public:
  bool empty() {
    return (map_.size() == 0);
  }

  size_t get_length() {
    return list_.size();
  }

  void clear() {
    std::lock_guard<std::mutex> lg(mutex_);
    map_.clear();
    list_.clear();
  }

  bool exists(std::shared_ptr<RecipeArgumentSpec>& key) {
    bool ret_flag{false};
    if (!empty() && map_.end() != map_.find(key)) {
      ret_flag = true;
    }
    return ret_flag;
  }

  std::pair<std::shared_ptr<RecipeArgumentSpec>, std::shared_ptr<RecipeHolder>>
      dropped_recipe;
  void add(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<RecipeHolder>& val);
  std::shared_ptr<RecipeHolder> get(std::shared_ptr<RecipeArgumentSpec>& key);
  bool drop_lru(size_t& num_recipes);
  void remove_oldest();
  void ResetDiskCache();
  void DeleteDiskCache();
  void FlushDiskCache();
  void Serialize();
  void Deserialize();

  static void SetHostMemoryThreshold(
      uint32_t host_memory_threshold = default_host_memory_threshold);

  size_t Size() const;
  size_t SynapseRecipeSize() const;
  static void DumpRecipeMemoryStat();
  static void DumpSynapseRecipeMemoryStat();
  static void DumpDynamicShapeMemoryStat();

  // friend std::ostream& operator<<(std::ostream& O, const RecipeCacheLRU& v);
  RecipeCacheLRU();
  ~RecipeCacheLRU() = default;

 private:
  RecipeCacheLRU(const RecipeCacheLRU&) = delete;
  RecipeCacheLRU& operator=(const RecipeCacheLRU&) = delete;
  bool drop_lru_impl(size_t& recipe_count, bool mem_exhausted = false);
  void insert(
      std::shared_ptr<RecipeArgumentSpec>& key,
      std::shared_ptr<RecipeHolder>& val);
  void InitDiskCache();

  std::mutex mutex_;
  static size_t max_size_;
  std::unique_ptr<DiskCache> disk_cache_;
  static const uint32_t default_host_memory_threshold = 90;

  std::list<std::pair<
      std::shared_ptr<RecipeArgumentSpec>,
      std::shared_ptr<RecipeHolder>>>
      list_;

  std::unordered_map<
      std::shared_ptr<RecipeArgumentSpec>,
      std::list<std::pair<
          std::shared_ptr<RecipeArgumentSpec>,
          std::shared_ptr<RecipeHolder>>>::iterator,
      RecipeArgumentSpecHash,
      RecipeArgumentSpecEqual>
      map_;

  serialization::RecipeCacheConfig recipe_cache_config_;
};
} // namespace habana
