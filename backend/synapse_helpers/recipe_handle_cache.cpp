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
#include "backend/synapse_helpers/recipe_handle_cache.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

namespace synapse_helpers {
recipe_handle_cache::recipe_handle_cache(device& device)
    : mutex_{}, device_{device}, evict_strategy_{NumberLimit(MAX_CACHE_SIZE)} {
  static_cast<void>(device_);
  enable_hit_count_ = (GET_ENV_FLAG_NEW(PT_HABANA_MAX_RECIPE_HIT_COUNT) != 0);
}

void recipe_handle_cache::insert(
    std::shared_ptr<recipe> recipe,
    const size_t key) {
  auto key_it = keys_access_list_.insert(keys_access_list_.begin(), key);
  cache_map_.emplace(key_wrapper_t(*key_it), std::make_pair(recipe, key_it));

  evict_strategy_.Added();
  while (evict_strategy_.IsEvictNeeded() && !cache_map_.empty()) {
    evict();
  }
}

void recipe_handle_cache::evict() {
  auto it = cache_map_.find(key_wrapper_t(keys_access_list_.back()));
  if (it != cache_map_.end()) {
    evict_strategy_.Removed();
    cache_map_.erase(it);
    keys_access_list_.pop_back();
  }
}

void recipe_handle_cache::drop() {
  cache_map_.clear();
  keys_access_list_.clear();
  evict_strategy_.Reset();
}

std::shared_ptr<recipe> recipe_handle_cache::get_recipe(
    const size_t key,
    synapse_helpers::graph& graph) {
  std::unique_lock<std::mutex> lck(mutex_);
  auto iter = cache_map_.find(key);
  if (iter != cache_map_.end()) {
    keys_access_list_.splice(
        keys_access_list_.begin(), keys_access_list_, iter->second.second);
    increaseHitCount_(key);
    return iter->second.first;
  } else {
    std::shared_ptr<recipe> r = std::make_shared<recipe>(device_);
    if (r->create(graph)) {
      insert(r, key);
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
    keys_access_list_.splice(
        keys_access_list_.begin(), keys_access_list_, iter->second.second);
    increaseHitCount_(key);
    return iter->second.first;
  }
  return nullptr;
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
  drop();
  clearHitCount();
}

} // namespace synapse_helpers
