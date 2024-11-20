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

#define PRINT_DEPRECATION_WARN_IF_ENV_USED(env_name) \
  if (IS_ENV_FLAG_DEFINED_NEW(env_name))             \
  std::clog << #env_name                             \
      " flag is going to be deprecated, use PT_HPU_RECIPE_CACHE_CONFIG instead\n"

RecipeCacheConfig::RecipeCacheConfig() {
  if (habana::GetRankFromEnv() == 0) {
    PRINT_DEPRECATION_WARN_IF_ENV_USED(PT_RECIPE_CACHE_PATH);
    PRINT_DEPRECATION_WARN_IF_ENV_USED(PT_CACHE_FOLDER_DELETE);
    PRINT_DEPRECATION_WARN_IF_ENV_USED(PT_CACHE_FOLDER_SIZE_MB);
  }

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
          auto rank_val = val;
          if (habana_helpers::IsInferenceMode()) {
            const char* s_rank = getenv("RANK") ? getenv("RANK") : "0";
            auto rank = std::atoi(s_rank);
            rank_val = rank_val + std::to_string(rank);
          }
          SET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH, rank_val.c_str(), 1);
        },
        [](std::string& val) {
          // parse if provided value is not empty
          // it it is empty then leave default
          if (!val.empty()) {
            auto value =
                PARSE_ENV_FLAG_NEW(PT_CACHE_FOLDER_DELETE, val.c_str());
            SET_ENV_FLAG_NEW(PT_CACHE_FOLDER_DELETE, value, 1);
          }
        },
        [](std::string& val) {
          // parse if provided value is not empty
          // it it is empty then leave default
          if (!val.empty()) {
            auto value =
                PARSE_ENV_FLAG_NEW(PT_CACHE_FOLDER_SIZE_MB, val.c_str());
            SET_ENV_FLAG_NEW(PT_CACHE_FOLDER_SIZE_MB, value, 1);
          }
        },
    };

    for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
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

const std::string& RecipeCacheConfig::path() const {
  return cache_directory_path_;
}

bool RecipeCacheConfig::delete_on_init() const {
  return delete_cache_on_init_;
}

void RecipeCacheConfig::disable_delete_on_init() {
  delete_cache_on_init_ = false;
  SET_ENV_FLAG_NEW(PT_CACHE_FOLDER_DELETE, false, 1);
}

unsigned int RecipeCacheConfig::cache_dir_max_size_mb() const {
  return cache_dir_max_size_mb_;
}
}; // namespace serialization