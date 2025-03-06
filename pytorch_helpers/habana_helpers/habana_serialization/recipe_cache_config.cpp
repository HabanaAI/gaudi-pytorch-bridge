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

#include "recipe_cache_config.h"

#include <absl/strings/match.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/misc_utils.h"

namespace serialization {

RecipeCacheConfig::RecipeCacheConfig() {
  reload();
}

void RecipeCacheConfig::reload() {
  // set to default values
  cache_directory_path_ = "";
  cache_dir_max_size_mb_ = 1024;
  delete_cache_on_init_ = false;

  if (!IS_ENV_FLAG_DEFINED_NEW(PT_HPU_RECIPE_CACHE_CONFIG)) {
    return;
  }

  std::string recipe_cache_config_var =
      GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  auto params = split_params(recipe_cache_config_var);
  HABANA_ASSERT(
      params.size() >= 1 && params.size() <= 3,
      "Expected number of parameters extracted from PT_HPU_RECIPE_CACHE_CONFIG should be from range <1:3>.");

  cache_directory_path_ = params[0];
  if (habana_helpers::IsInferenceMode()) {
    const char* s_rank = getenv("RANK") ? getenv("RANK") : "0";
    auto rank = std::atoi(s_rank);
    cache_directory_path_ += std::to_string(rank);
  }

  if (params.size() >= 2 && !params[1].empty()) {
    delete_cache_on_init_ = parse_env_by_type<bool>(
        "PT_HPU_RECIPE_CACHE_CONFIG", params[1].c_str());
  }

  if (params.size() >= 3 && !params[2].empty()) {
    cache_dir_max_size_mb_ = parse_env_by_type<unsigned>(
        "PT_HPU_RECIPE_CACHE_CONFIG", params[2].c_str());
  }
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

const std::string& RecipeCacheConfig::path() const {
  return cache_directory_path_;
}

bool RecipeCacheConfig::delete_on_init() const {
  return delete_cache_on_init_;
}

void RecipeCacheConfig::disable_delete_on_init() {
  delete_cache_on_init_ = false;
  std::string new_config = cache_directory_path_ + ",false," +
      std::to_string(cache_dir_max_size_mb_);
  SET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG, new_config.c_str(), 1);
}

unsigned int RecipeCacheConfig::cache_dir_max_size_mb() const {
  return cache_dir_max_size_mb_;
}
}; // namespace serialization