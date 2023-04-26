/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "inter_host_cache.h"

namespace serialization {

class RecipeCache {
 public:
  RecipeCache(std::string cache_path);
  ~RecipeCache();

  // if synRecipeHandle is nullptr, it will not be serialized to file, only
  // metadata
  void store(
      std::string cache_id,
      std::shared_ptr<synapse_helpers::graph::recipe_handle> const&
          recipeHandle,
      std::stringstream&& metadata);
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

 private:
  std::mutex mut_;
  std::condition_variable cond_var_;
  std::string cache_path_;
  bool is_cache_valid_;
  std::unique_ptr<InterHostCache> inter_host_cache_;
  std::shared_ptr<CacheFileHandler> cfHandler;
  std::future<void> send_thread;

  // map to track opened metadata files, so can be closed, once cache entry is
  // stored on disk
  std::unordered_map<std::string, int> meta2fd_map_;
};

} // namespace serialization
