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
#include "backend/kernel/hpu_recipe_cache.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "habana_serialization/cache_version.h"
#include "habana_serialization/recipe_cache_config.h"

namespace habana {

// static initializations

size_t RecipeCacheLRU::max_size_ = PGM_LRU_MAX_LAZY_NRECIPES;

size_t RecipeValueSpec::count = 0;
size_t RecipeValueSpec::recipe_count = 0;
size_t RecipeValueSpec::dynamic_recipe_count = 0;
size_t RecipeValueSpec::total_recipe_ntbytes = 0;
size_t RecipeValueSpec::compile_count = 0;

size_t RecipeArgumentSpecHash::operator()(
    const std::shared_ptr<RecipeArgumentSpec>& v) const {
  return v->hashCode();
}

// Comparator for RecipeArgumentSpec
bool RecipeArgumentSpecEqual::operator()(
    const std::shared_ptr<RecipeArgumentSpec>& v1,
    const std::shared_ptr<RecipeArgumentSpec>& v2) const {
  if (nullptr == v1 && nullptr == v2)
    return true;
  if (nullptr == v1)
    return false;
  if (nullptr == v2)
    return false;

  return (*v1) == (*v2);
}

void RecipeCacheLRU::add(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<RecipeHolder>& val) {
  std::lock_guard<std::mutex> lg(mutex_);

  insert(key, val);

  if (disk_cache_) {
    disk_cache_->Add(*val, *key);
  }
}

void RecipeCacheLRU::insert(
    std::shared_ptr<RecipeArgumentSpec>& key,
    std::shared_ptr<RecipeHolder>& val) {
  TORCH_CHECK(
      map_.size() == list_.size(),
      "lru cache corruption, map size ",
      map_.size(),
      " not equal to list_size ",
      list_.size());

  size_t rcnt{0};
  bool dropped{true};
  if (!val->rvs_->dynamic_graph) {
    while (
        !map_.empty() && dropped &&
        (map_.size() >= max_size_ || habana::IsHostMemoryThresholdReached())) {
      dropped = drop_lru_impl(rcnt);
      if (!dropped) {
        PT_BRIDGE_DEBUG(
            "all recipes are in use, could not drop any, current recipe count ",
            rcnt);
      }
    }
  }

  val->rvs_->increment_recipe_count();

  auto mit = map_.find(key);
  if (mit != map_.end()) {
    PT_BRIDGE_DEBUG(
        "problematic key ",
        key->hashCode(),
        " another recipe already exists in cache");
  } else {
    list_.push_front(std::pair<
                     std::shared_ptr<RecipeArgumentSpec>,
                     std::shared_ptr<RecipeHolder>>(key, val));
    map_.emplace(key, list_.begin());

    RecipeValueSpec::total_recipe_ntbytes += val->rl_->ntensorbytes_;
  }
}

std::shared_ptr<RecipeHolder> RecipeCacheLRU::get(
    std::shared_ptr<RecipeArgumentSpec>& key) {
  std::lock_guard<std::mutex> lg(mutex_);
  if (exists(key)) {
    TORCH_CHECK(
        map_.size() == list_.size(),
        "lru cache corruption, map size ",
        map_.size(),
        " not equal to list_size ",
        list_.size());

    TORCH_CHECK(exists(key), "Recipe does not exist in map");

    auto mit = map_.find(key);
    list_.splice(list_.begin(), list_, mit->second);

    // increment the use_count so that the recipe is not removed from cache
    // it is the responsibility of the caller of get function to
    // decrement the use count after the execution is completed

    return list_.front().second;
  } else if (disk_cache_) {
    auto val = disk_cache_->Find(*key);
    if (val) {
      PT_BRIDGE_DEBUG(
          "recipe was not found in LRU cache, but was found on disk, key:",
          key->hashCode());
      insert(key, val);
      return val;
    }
  }
  return {nullptr};
}

bool RecipeCacheLRU::drop_lru(size_t& num_recipes) {
  std::lock_guard<std::mutex> lg(mutex_);
  bool dropped = drop_lru_impl(num_recipes, true);
  return dropped;
}

bool RecipeCacheLRU::drop_lru_impl(size_t& num_recipes, bool mem_exhausted) {
  bool dropped{false};
  int use_count = 0;
  // remove a recipe from the last that is not being used
  if (!map_.empty()) {
    auto lit = list_.end();
    lit--;

    while (lit->second->is_in_use() && lit != list_.begin()) {
      PT_BRIDGE_DEBUG(
          "recipe is in use, key ",
          lit->first->hashCode(),
          ", size ",
          synapse_helpers::get_mem_str(lit->second->rl_->ntensorbytes_));
      lit--;
    }

    // delete the recipe only if it is not in use
    // otherwise the caller need to wait
    if (!lit->second->is_in_use()) {
      if (mem_exhausted) {
        PT_BRIDGE_DEBUG(
            "memory exhausted : removing recipe, key ",
            lit->first->hashCode(),
            ", size ",
            synapse_helpers::get_mem_str(lit->second->rl_->ntensorbytes_));
      } else {
        PT_BRIDGE_DEBUG(
            "lru max size ",
            max_size_,
            " reached : removing recipe, key ",
            lit->first->hashCode(),
            ", size ",
            synapse_helpers::get_mem_str(lit->second->rl_->ntensorbytes_));
      }

      lit->second->rvs_->decrement_recipe_count();
      RecipeValueSpec::total_recipe_ntbytes -= lit->second->rl_->ntensorbytes_;

      // Drop the entry from map_ and list_
      dropped_recipe.first = lit->first;
      dropped_recipe.second = lit->second;
      map_.erase(lit->first);
      list_.erase(lit);
      dropped = true;

      PT_BRIDGE_DEBUG(
          "after dropping lru recipe, #recipes ",
          RecipeValueSpec::get_recipe_count(),
          ", total size of graph recipes ",
          synapse_helpers::get_mem_str(RecipeValueSpec::total_recipe_ntbytes));
    } else {
      use_count++;
      PT_BRIDGE_DEBUG(
          "all recipes are in use used_recipe_count=",
          use_count,
          " can not drop any recipe");
    }
  }

  num_recipes = map_.size() - use_count;
  return dropped;
}

RecipeCacheLRU::RecipeCacheLRU() {
  InitDiskCache();

  max_size_ = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1
      ? PGM_LRU_MAX_LAZY_NRECIPES
      : PGM_LRU_MAX_EAGER_NRECIPES;

  char* smaxsize = getenv("HABANA_PGM_LRU_MAX");
  if (smaxsize != nullptr) {
    max_size_ = std::max(PGM_LRU_MIN_NRECIPES, atoi(smaxsize));
  }
}

void RecipeCacheLRU::InitDiskCache() {
  // Set disk_cache_ if PT_RECIPE_CACHE_PATH is defined
  if (!recipe_cache_config_.path().empty())
    disk_cache_ = absl::make_unique<DiskCache>(recipe_cache_config_);
}

void RecipeCacheLRU::ResetDiskCache() {
  recipe_cache_config_.reload();
  InitDiskCache();
}

void RecipeCacheLRU::DeleteDiskCache() {
  if (disk_cache_) {
    disk_cache_.reset();
  }
}

void RecipeCacheLRU::FlushDiskCache() {
  if (disk_cache_) {
    disk_cache_->flush();
  }
}

void RecipeCacheLRU::SetHostMemoryThreshold(uint32_t host_memory_threshold) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT)) {
    SET_ENV_FLAG_NEW(
        PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT, host_memory_threshold, 1);
  }
}

size_t RecipeCacheLRU::Size() const {
  size_t size = 0;
  for (auto const& [recipeArgumentSpec, RecipeHolder] : list_) {
    size += recipeArgumentSpec->Size();
    size += RecipeHolder->Size();
  }
  return size;
}

void RecipeCacheLRU::Serialize() {
  if (recipe_cache_config_.path().empty()) {
    PT_BRIDGE_DEBUG("disk recipe cache not path, cannot serialize");
    return;
  }

  disk_cache_ = absl::make_unique<DiskCache>(recipe_cache_config_);

  for (const auto& ele : list_) {
    auto val = disk_cache_->Find(*(ele.first));
    if (val == nullptr) {
      disk_cache_->Add(*(ele.second), *(ele.first));
    }
  }
}

void RecipeCacheLRU::Deserialize() {
  if (recipe_cache_config_.path().empty()) {
    PT_BRIDGE_DEBUG("disk recipe cache not path, cannot De-serialize");
    return;
  }

  recipe_cache_config_.disable_delete_on_init();
  disk_cache_ = absl::make_unique<DiskCache>(recipe_cache_config_);
}

size_t RecipeCacheLRU::SynapseRecipeSize() const {
  size_t size = 0;
  for (auto const& p : list_) {
    size += p.second->rl_->recipe_->get_recipe_host_mem_size();
  }
  return size;
}

DiskCache::DiskCache(
    const serialization::RecipeCacheConfig& recipe_cache_config)
    : recipe_cache_(recipe_cache_config)
// TODO: add pytorch version, TICKET SW-62210
{
  if (GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_IGNORE_VERSION)) {
    cache_id_suffix_ = "";
  } else {
    constexpr int bufferSize = 256;
    char versionStr[bufferSize];
    synDriverGetVersion(versionStr, bufferSize);
    cache_id_suffix_ =
        absl::StrCat("_", CacheVersion::libs_env_hash(), "_syn", versionStr);
  }
}

void DiskCache::Add(
    const RecipeHolder& val,
    const RecipeArgumentSpec& argSpec) // NOLINT
{
  std::stringstream ss;
  val.Serialize(ss);
  auto hashCode = std::to_string(argSpec.hashCode());
  recipe_cache_.store(hashCode + cache_id_suffix_, val.rl_->recipe_, ss);
  if (val.rl_->recipe_ && !val.rl_->recipe_->recipe_name_.empty()) {
    PT_BRIDGE_DEBUG(
        "Storing in disc cache: recipe:key: ",
        val.rl_->recipe_->recipe_name_,
        ":",
        hashCode);
  } else {
    PT_BRIDGE_DEBUG("Storing only metadata in disc cache, key: ", hashCode);
  }

  static const auto dump_debug_info =
      GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_DUMP_DEBUG);
  if (dump_debug_info) {
    static int debug_id = 0;
    std::string recipe_name = val.rl_->recipe_
        ? val.rl_->recipe_->recipe_name_
        : "recipe " + std::to_string(debug_id++);
    std::string hash_content_filepath = recipe_cache_.get_cache_path() + "/" +
        hashCode + cache_id_suffix_ + "_" + recipe_name + ".hash_content";
    std::ofstream hash_content_file(hash_content_filepath.c_str());
    if (!hash_content_file.is_open()) {
      LOG(FATAL) << "Failed to open hash content file for writing...";
    }
    hash_content_file << argSpec;
    hash_content_file.close();
  }
}

void DiskCache::flush() {
  recipe_cache_.flush();
}

std::shared_ptr<RecipeHolder> DiskCache::Find(const RecipeArgumentSpec& spec) {
  std::stringstream ss;
  auto res = recipe_cache_.lookup(
      std::to_string(spec.hashCode()) + cache_id_suffix_, ss);
  if (res) {
    auto val = std::make_shared<RecipeHolder>(ss);
    if (*res != nullptr) {
      if (!val->rl_->recipe_) {
        PT_BRIDGE_WARN(
            "Unexpected nullptr recipe came from cache entry for hash ",
            std::to_string(spec.hashCode()));
        return nullptr;
      }
      val->rl_->recipe_->syn_recipe_handle_ = *res;
      val->rl_->recipe_->in_execution_phase_ = true;
    }
    return val;
  }
  return nullptr;
}

} // namespace habana
