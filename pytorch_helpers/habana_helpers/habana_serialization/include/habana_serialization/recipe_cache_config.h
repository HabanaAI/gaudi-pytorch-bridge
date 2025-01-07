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

#include <memory>
#include <mutex>
#include <string>
#include <vector>
namespace serialization {

class RecipeCacheConfig {
 public:
  RecipeCacheConfig();
  void reload();
  const std::string& path() const;
  bool delete_on_init() const;
  void disable_delete_on_init();
  unsigned cache_dir_max_size_mb() const;

 private:
  std::vector<std::string> split_params(const std::string& config);
  std::string cache_directory_path_;
  bool delete_cache_on_init_;
  unsigned int cache_dir_max_size_mb_;
};
} // namespace serialization