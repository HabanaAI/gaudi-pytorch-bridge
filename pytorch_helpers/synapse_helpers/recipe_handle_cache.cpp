
/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/recipe_handle_cache.h"
#include "synapse_helpers/env_utils.h"

constexpr std::size_t MAX_CACHE_SIZE = 10000;

namespace synapse_helpers {
recipe_handle_cache::recipe_handle_cache(device& device)
    : mutex_{}, device_{device} {}

std::shared_ptr<recipe> recipe_handle_cache::get_recipe(
    const size_t key,
    synapse_helpers::graph& graph) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto iter = cache_map_.find(key);
  if (iter != cache_map_.end()) {
    return iter->second;
  } else {
    std::shared_ptr<recipe> r = std::make_shared<recipe>();
    if (r->create(graph)) {
      cache_map_[key] = r;
      return r;
    }
  }
  return nullptr;
}

std::shared_ptr<recipe> recipe_handle_cache::get_recipe(size_t key) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto iter = cache_map_.find(key);
  if (iter != cache_map_.end()) {
    return iter->second;
  }
  return nullptr;
}

void recipe_handle_cache::remove_recipe(const size_t key) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto iter = cache_map_.find(key);
  if (iter != cache_map_.end()) {
    cache_map_.erase(iter);
  }
}

bool recipe_handle_cache::isCached(size_t hash) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto iter = cache_map_.find(hash);
  if (!cache_map_.empty() && iter != cache_map_.end()) {
    return true;
  }
  return false;
}

recipe_handle_cache::~recipe_handle_cache() {
  cache_map_.clear();
}

bool IsCachingEnabled() {
  return synapse_helpers::get_bool_env_var("PT_ENABLE_HABANA_CACHING", true);
}
} // namespace synapse_helpers
