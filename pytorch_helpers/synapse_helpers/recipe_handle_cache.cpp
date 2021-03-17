
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
#include "habana_helpers/logging.h"
#include "synapse_helpers/env_flags.h"

namespace synapse_helpers {
recipe_handle_cache::recipe_handle_cache(device& device)
    : mutex_{}, device_{device} {
  static_cast<void>(device_);
  enable_hit_count_ = (GET_ENV_FLAG(PT_HABANA_MAX_RECIPE_HIT_COUNT) != 0);
}

std::shared_ptr<recipe> recipe_handle_cache::get_recipe(
    const size_t key,
    synapse_helpers::graph& graph) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto iter = cache_map_.find(key);
  if (iter != cache_map_.end()) {
    increaseHitCount_(key);
    return iter->second;
  } else {
    std::shared_ptr<recipe> r = std::make_shared<recipe>();
    if (r->create(graph)) {
      cache_map_[key] = r;
      increaseHitCount_(key);
      return r;
    }
  }
  return nullptr;
}

std::shared_ptr<recipe> recipe_handle_cache::get_recipe(size_t key) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto iter = cache_map_.find(key);
  if (iter != cache_map_.end()) {
    increaseHitCount_(key);
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

size_t recipe_handle_cache::getCount() {
  std::unique_lock<std::mutex> lck(mutex_);
  return cache_map_.size();
}

void recipe_handle_cache::increaseHitCount(const size_t key) {
  if (!enable_hit_count_)
    return;

  std::unique_lock<std::mutex> lck(mutex_);
  increaseHitCount_(key);
}

void recipe_handle_cache::increaseHitCount_(const size_t key) {
  if (!enable_hit_count_)
    return;

  if (0 == getHitCount(key)) {
    hit_counter_[key] = 0;
  }

  hit_counter_[key] += 1;
}

int recipe_handle_cache::getActiveRecipeCount() {
  if (!enable_hit_count_)
    return -1;

  std::unique_lock<std::mutex> lck(mutex_);
  return int(hit_counter_.size());
}

int recipe_handle_cache::getHitCount(const size_t key) {
  if (!enable_hit_count_)
    return -1;

  std::unique_lock<std::mutex> lck(mutex_);
  return (hit_counter_.count(key) ? hit_counter_[key] : 0);
}

void recipe_handle_cache::printHitCount() {
  if (!enable_hit_count_)
    return;

  std::unique_lock<std::mutex> lck(mutex_);
  PT_SYNHELPER_DEBUG("Number of active recipes ", hit_counter_.size());
  for (auto p : hit_counter_) {
    PT_SYNHELPER_DEBUG("Recipe key ", p.first, ", #hits ", p.second);
  }
}

void recipe_handle_cache::clearHitCount() {
  if (!enable_hit_count_)
    return;

  std::unique_lock<std::mutex> lck(mutex_);
  hit_counter_.clear();
}

recipe_handle_cache::~recipe_handle_cache() {
  cache_map_.clear();
  clearHitCount();
}

} // namespace synapse_helpers
