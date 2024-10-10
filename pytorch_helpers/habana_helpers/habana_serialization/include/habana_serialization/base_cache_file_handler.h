/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
  BaseCacheFileHandler();
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