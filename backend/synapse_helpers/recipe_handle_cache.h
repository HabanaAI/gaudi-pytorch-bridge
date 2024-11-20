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
#include <absl/container/flat_hash_set.h>
#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_helpers/logging.h"

#pragma once

namespace synapse_helpers {
class device;
class recipe;

class NumberLimit {
 public:
  NumberLimit(uint64_t max_number) : number_limit_(max_number){};
  bool IsEvictNeeded() {
    return used_ > number_limit_;
  }
  void Added() {
    ++used_;
  };
  void Removed() {
    --used_;
  };
  void Reset() {
    used_ = 0;
  };
  void Print() {
    PT_SYNHELPER_DEBUG("Recipe used count:: ", used_);
  }

 private:
  const uint64_t number_limit_;
  uint64_t used_ = 0;
};

constexpr std::size_t MAX_CACHE_SIZE = 9000;

class recipe_handle_cache {
 public:
  explicit recipe_handle_cache(device& device);
  recipe_handle_cache(const recipe_handle_cache&) = delete;
  recipe_handle_cache(recipe_handle_cache&&) = delete;
  recipe_handle_cache& operator=(const recipe_handle_cache&) = delete;
  recipe_handle_cache& operator=(recipe_handle_cache&&) = delete;

  ~recipe_handle_cache();

  std::shared_ptr<recipe> get_recipe(
      const size_t key,
      synapse_helpers::graph& graph);
  std::shared_ptr<recipe> get_recipe(const size_t key);
  bool isCached(size_t key);

  size_t getCount();
  void increaseHitCount(const size_t key);
  int getActiveRecipeCount();
  int getHitCount(const size_t key);
  void clearHitCount();
  void printHitCount();

 private:
  std::mutex mutex_;
  device& device_;
  NumberLimit evict_strategy_;
  bool enable_hit_count_{false};
  std::unordered_map<size_t, int> hit_counter_;
  void increaseHitCount_(const size_t key);

  struct key_equal {
    bool operator()(const size_t& a, const size_t& b) const {
      return a == b;
    }
  };

  struct Hash {
    size_t operator()(const size_t& v) const {
      return std::hash<size_t>{}(v);
    }
  };

  using keys_access_list_t = std::list<size_t>;
  using key_wrapper_t = std::reference_wrapper<const size_t>;

  // keys_access_list_ is a list which has keys ordered by their access time.
  // The most recently used key is in front of the list. clusters_map_ also
  // containe iterator to key in a keys access list in order to move key for
  // recently used item into beginning of that list. Hash map stores keys as
  // pointer to a list in order to reduce memory consumption.
  keys_access_list_t keys_access_list_;
  absl::flat_hash_map<
      key_wrapper_t,
      std::pair<std::shared_ptr<recipe>, keys_access_list_t::iterator>,
      Hash,
      std::equal_to<size_t>>
      cache_map_;

  void insert(std::shared_ptr<recipe> recipe, const size_t key);
  void evict();
  void drop();
};

} // namespace synapse_helpers
