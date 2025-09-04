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

#include <synapse_api_types.h>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include "backend/synapse_helpers/graph.h"
#include "cache_file_handler.h"
#include "habana_helpers/job_thread.h"
#include "recipe_cache_config.h"

namespace serialization {

class RecipeCache {
 public:
  RecipeCache(const RecipeCacheConfig& recipe_cache_config);
  ~RecipeCache();

  // if synRecipeHandle is nullptr, it will not be serialized to file, only
  // metadata
  void store(
      std::string cache_id,
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_handle,
      const std::stringstream& metadata);
  // if operation is successful (optional not empty), metadata will be populated
  // synRecipeHandle optional can be set to nullptr, that means, the cache entry
  // only had metadata
  absl::optional<synRecipeHandle> lookup(
      std::string cache_id,
      std::ostream& metadata);
  // Location of stored recipes
  std::string get_cache_path() const {
    return cache_path_;
  }
  void flush();

 private:
  // lockfree_lookup/lockfree_store_task are mirror functions of
  // lookup/store_task that perform serialization and lookup w/o depending on
  // file locks. The idea is to create cache entries unique to a given process
  // ID and machine (MAC address). Such cache entries can be safely written to,
  // and can be tried for reading. Any cache lookup will spawn
  // TEMP_FILE_PREFIX_<original_name> metadata file, that will indicate to other
  // workers, that this process is compiling. In case other is looking for
  // a file and finds 'compiling' file, it will wait (timeout:
  // PT_HPU_RECIPE_CACHE_NFS_TIMEOUT_S) and try to read non compiling file.
  absl::optional<synRecipeHandle> lockfree_lookup(
      std::string cache_id,
      std::ostream& metadata);
  void lockfree_store_task(
      const std::string& cache_id,
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipeHandle,
      const std::string& metadata);

  std::string cache_path_;
  bool is_cache_valid_{false};
  std::shared_ptr<CacheFileHandler> cf_handler_;
  bool cache_on_nfs_;
  std::unique_ptr<habana_helpers::JobThread> cache_thread_;
  void store_task(
      const std::string& cache_id,
      std::shared_ptr<synapse_helpers::graph::recipe_handle> recipeHandle,
      const std::string& metadata);
};

} // namespace serialization
