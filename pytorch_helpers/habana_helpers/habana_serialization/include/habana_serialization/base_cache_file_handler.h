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

#include <fcntl.h>

#include <map>
#include <optional>
#include <vector>

#if !defined __GNUC__ || __GNUC__ >= 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "cache_file_handler.h"

namespace serialization {

class BaseCacheFileHandler : public CacheFileHandler {
 public:
  BaseCacheFileHandler(const RecipeCacheConfig& recipe_cache_config);
  virtual ~BaseCacheFileHandler();

 protected:
  void checkAndDelete() override;

 private:
  struct RecipeInfo {
    std::string recipe_id;
    uint64_t recipe_size;
    uint64_t metadata_size;
    fs::file_time_type created;
  };

  bool acquire_access_for_eviction(bool block = false);
  void release_access_for_eviction();
  void evict_recipe_if_needed();
  std::vector<RecipeInfo> get_recipes_list_by_date();
  uint64_t calculate_recipes_total_size(std::vector<RecipeInfo>& recipes);
  bool delete_recipe(RecipeInfo& r_info);

  int eviction_lock_fd_;
};

} // namespace serialization