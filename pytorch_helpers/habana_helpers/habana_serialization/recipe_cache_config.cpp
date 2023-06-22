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

#include "recipe_cache_config.h"

#include <absl/strings/match.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

namespace serialization {

#define OVERRIDE_ENV_USING_STR(env_name, val) \
  setenv(#env_name, val.c_str(), 1);          \
  GET_ENV_FLAG_NEW(env_name, 1)

RecipeCacheConfig::RecipeCacheConfig() {
  reload();
}

void RecipeCacheConfig::reload() {
  // If PT_HPU_RECIPE_CACHE_CONFIG is set then PT_RECIPE_CACHE_PATH,
  // PT_CACHE_FOLDER_DELETE, and PT_CACHE_FOLDER_SIZE_MB are overriden by
  // parameters read from PT_HPU_RECIPE_CACHE_CONFIG
  if (IS_ENV_FLAG_DEFINED_NEW(PT_HPU_RECIPE_CACHE_CONFIG)) {
    std::string recipe_cache_config_var =
        GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);

    auto params = split_params(recipe_cache_config_var);
    HABANA_ASSERT(
        params.size() >= 1 && params.size() <= 3,
        "Expected number of parameters extracted from PT_HPU_RECIPE_CACHE_CONFIG should be from range <1:3>.");

    const std::array<std::function<void(std::string&)>, 3> env_var_setters = {
        [](std::string& val) {
          OVERRIDE_ENV_USING_STR(PT_RECIPE_CACHE_PATH, val);
        },
        [](std::string& val) {
          OVERRIDE_ENV_USING_STR(PT_CACHE_FOLDER_DELETE, val);
        },
        [](std::string& val) {
          OVERRIDE_ENV_USING_STR(PT_CACHE_FOLDER_SIZE_MB, val);
        },
    };

    for (int param_idx = 0; param_idx < params.size(); param_idx++) {
      env_var_setters[param_idx](params[param_idx]);
    }
  }

  cache_directory_path_ = GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH);
  cache_dir_max_size_mb_ = GET_ENV_FLAG_NEW(PT_CACHE_FOLDER_SIZE_MB);
  delete_cache_on_init_ = GET_ENV_FLAG_NEW(PT_CACHE_FOLDER_DELETE);
}

std::vector<std::string> RecipeCacheConfig::split_params(
    const std::string& config) {
  std::istringstream iss(config);
  std::string param;
  std::vector<std::string> params;
  while (std::getline(iss, param, ',')) {
    params.push_back(param);
  }
  return params;
}

const std::string& RecipeCacheConfig::path() {
  return cache_directory_path_;
}

bool RecipeCacheConfig::delete_on_init() {
  return delete_cache_on_init_;
}

void RecipeCacheConfig::disable_delete_on_init() {
  delete_cache_on_init_ = false;
  SET_ENV_FLAG_NEW(PT_CACHE_FOLDER_DELETE, false, 1);
}

unsigned int RecipeCacheConfig::cache_dir_max_size_mb() {
  return cache_dir_max_size_mb_;
}

std::mutex RecipeCacheConfig::mutex_;

}; // namespace serialization