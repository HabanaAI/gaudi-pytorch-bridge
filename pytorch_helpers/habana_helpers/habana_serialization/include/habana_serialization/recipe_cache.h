/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <synapse_api_types.h>
#include <synapse_helpers/graph.h>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
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

  // helper function to find fd for a given metada file and erase it from the
  // map
  int pop_meta_fd(const std::string& lock_file);
};

} // namespace serialization
