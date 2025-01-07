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

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <sstream>

#include "base_cache_file_handler.h"
#include "cache_file_handler.h"
#include "habana_helpers/logging.h"

#define CACHEFILE_LOG "[CACHEFILE] "

namespace serialization {

BaseCacheFileHandler::BaseCacheFileHandler(
    const RecipeCacheConfig& recipe_cache_config)
    : CacheFileHandler(recipe_cache_config), eviction_lock_fd_(-1) {}

BaseCacheFileHandler::~BaseCacheFileHandler() {
  if (eviction_lock_fd_ >= 0) {
    fileClose(eviction_lock_fd_);
    eviction_lock_fd_ = -1;
  }
}

void BaseCacheFileHandler::checkAndDelete() {
  evict_recipe_if_needed();
}

std::vector<BaseCacheFileHandler::RecipeInfo> BaseCacheFileHandler::
    get_recipes_list_by_date() {
  fs::path disk_cache_path{getCachePath()};
  std::vector<BaseCacheFileHandler::RecipeInfo> recipes_in_cache;
  std::set<std::string> recipe_ids;

  try {
    if (!fs::exists(disk_cache_path)) {
      return {};
    }

    auto de = fs::directory_iterator{disk_cache_path};
    while (de != fs::end(de)) {
      auto file_name = de->path().filename().string();
      std::string recipe_id = file_name.substr(0, file_name.rfind("."));
      auto r_path = recipe_file_path(disk_cache_path, recipe_id);
      auto met_path = metadata_file_path(disk_cache_path, recipe_id);

      if (fs::exists(r_path) && fs::exists(met_path) &&
          recipe_ids.count(recipe_id) == 0) {
        recipe_ids.insert(recipe_id);

        RecipeInfo r_info = {
            recipe_id,
            fs::file_size(r_path),
            fs::file_size(met_path),
            fs::last_write_time(r_path)};

        recipes_in_cache.push_back(r_info);

        PT_HABHELPER_DEBUG(
            CACHEFILE_LOG,
            "recipe_id: ",
            recipe_id,
            ", rec size: ",
            r_info.recipe_size,
            ", met size: ",
            r_info.metadata_size);
      }
      de++;
    }
  } catch (fs::filesystem_error& err) {
    PT_HABHELPER_WARN(
        CACHEFILE_LOG,
        "Can't calculate current disk cache space consumption. Disk cache eviction may not work correctly.");
    return {};
  }

  std::sort(
      recipes_in_cache.begin(), recipes_in_cache.end(), [](auto& a, auto& b) {
        return a.created < b.created;
      });

  return recipes_in_cache;
}

uint64_t BaseCacheFileHandler::calculate_recipes_total_size(
    std::vector<RecipeInfo>& recipes) {
  uint64_t recipes_size = 0;

  for (auto& r_info : recipes) {
    recipes_size += r_info.recipe_size;
    recipes_size += r_info.metadata_size;
  }
  return recipes_size;
}

void BaseCacheFileHandler::evict_recipe_if_needed() {
  std::optional<uint64_t> recipe_cache_dir_max_size = getMaxFolderSize();
  fs::path disk_cache_path{getCachePath()};

  if (!recipe_cache_dir_max_size.has_value()) {
    // eviction is disabled
    PT_HABHELPER_DEBUG(CACHEFILE_LOG, "Disk cache eviction is disabled.");
    return;
  }

  if (!acquire_access_for_eviction(true)) {
    PT_HABHELPER_DEBUG(CACHEFILE_LOG, "No access for eviction.");
    return;
  }

  auto recipes_list = get_recipes_list_by_date();
  uint64_t recipes_total_size = calculate_recipes_total_size(recipes_list);
  PT_HABHELPER_DEBUG(
      CACHEFILE_LOG,
      "Recipes total size: ",
      recipes_total_size,
      ", limit: ",
      recipe_cache_dir_max_size.value());

  for (auto& r_info : recipes_list) {
    if (recipes_total_size <= recipe_cache_dir_max_size.value()) {
      break;
    }
    PT_HABHELPER_DEBUG(
        CACHEFILE_LOG,
        "Recipes total size exceeded the limit: ",
        recipes_total_size,
        "/",
        recipe_cache_dir_max_size.value(),
        ". Trying to remove: ",
        r_info.recipe_id);
    bool deleted = delete_recipe(r_info);
    if (deleted) {
      recipes_total_size -= (r_info.recipe_size + r_info.metadata_size);
      PT_HABHELPER_INFO(
          CACHEFILE_LOG,
          "Removed ",
          r_info.recipe_id,
          "successfully. Disk cache size after removal: ",
          recipes_total_size);
    }
  }

  release_access_for_eviction();

  if (recipes_total_size <= recipe_cache_dir_max_size.value()) {
    // Because recipe deleting may fail and this is normal case to handle
    // Show a warning instead of assertion error
    PT_HABHELPER_WARN(
        CACHEFILE_LOG,
        "Recipes total size still exceeded the limit after eviction: ",
        recipes_total_size,
        "/",
        recipe_cache_dir_max_size.value(),
        ".");
  }
}

bool BaseCacheFileHandler::delete_recipe(RecipeInfo& r_info) {
  fs::path disk_cache_path{getCachePath()};
  fs::path r_path{recipe_file_path(disk_cache_path, r_info.recipe_id)};
  fs::path met_path{metadata_file_path(disk_cache_path, r_info.recipe_id)};

  size_t size = 0;
  bool removed_successfully = false;

  int fd_r = openAndLockFile(r_path.string(), O_RDONLY, true, size);
  int fd_met = openAndLockFile(met_path.string(), O_RDONLY, true, size);

  if (fd_r < 0 || fd_met < 0) {
    PT_HABHELPER_WARN(
        CACHEFILE_LOG,
        "Couldn't open and lock recipe or metadata file - fd_r = ",
        fd_r,
        ", fd_met = ",
        fd_met);
  } else {
    try {
      fs::remove(r_path);
      fs::remove(met_path);
      PT_HABHELPER_DEBUG(CACHEFILE_LOG, "Deleted ", r_info.recipe_id);
      removed_successfully = true;
    } catch (fs::filesystem_error& err) {
      PT_HABHELPER_WARN(
          CACHEFILE_LOG,
          "File system error during removing recipe/metadata for ",
          r_info.recipe_id);
    }
  }
  return removed_successfully;
}

bool BaseCacheFileHandler::acquire_access_for_eviction(bool block) {
  size_t size = 0;
  fs::path cache_dir_path{getCachePath()};
  fs::path eviction_lock_file_path = cache_dir_path / "eviction.lock";

  eviction_lock_fd_ = openAndLockFile(
      eviction_lock_file_path.string(), O_RDWR | O_CREAT, block, size);

  if (eviction_lock_fd_ >= 0) {
    PT_HABHELPER_DEBUG(CACHEFILE_LOG, "Locked eviction directory successfully");
    return true;
  } else {
    return false;
  }
}

void BaseCacheFileHandler::release_access_for_eviction() {
  fs::path cache_dir_path{getCachePath()};
  fs::path eviction_lock_file_path = cache_dir_path / "eviction.lock";

  if (eviction_lock_fd_ >= 0) {
    PT_HABHELPER_DEBUG(CACHEFILE_LOG, "Releasing lock");
    fileClose(eviction_lock_fd_);
    eviction_lock_fd_ = -1;
  }
}

} // namespace serialization
