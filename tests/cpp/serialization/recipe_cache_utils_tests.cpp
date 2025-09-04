/**
 * Copyright (c) 2025 Intel Corporation
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
#include <gtest/gtest.h>
#include "pytorch_helpers/habana_helpers/habana_serialization/include/habana_serialization/cache_file_handler.h"
#include "pytorch_helpers/habana_helpers/habana_serialization/include/habana_serialization/cache_version.h"

using namespace serialization;

TEST(RecipeCacheUtils, recipe_file_path) {
  std::string path_to_cache = "/asd/vwd/edf";
  std::string cache_id = "unique_id";
  ASSERT_EQ(
      recipe_file_path(path_to_cache, cache_id),
      path_to_cache + "/" + cache_id + RECIPE_SUFFIX);
}

TEST(RecipeCacheUtils, metadata_file_path) {
  std::string path_to_cache = "/asd/vwd/edf";
  std::string cache_id = "unique_id";
  ASSERT_EQ(
      metadata_file_path(path_to_cache, cache_id),
      path_to_cache + "/" + cache_id + METADATA_SUFFIX);
}

TEST(RecipeCacheUtils, insert_temp_prefix_filename) {
  std::string test_path = "/asd/vwd/edf/filename";
  ASSERT_EQ(
      insert_temp_prefix_filename(test_path), "/asd/vwd/edf/temp_filename");
}

TEST(RecipeCacheUtils, combined_pid_mac_addr) {
  auto pid_mac = CacheVersion::combined_pid_mac_addr();

  // starts with PID
  ASSERT_TRUE(pid_mac.find(std::to_string(getpid())) == 0);
  // MAC address should always be valid
  ASSERT_NE(pid_mac, std::to_string(getpid()) + "_000000000000");
  // length is PID + 1 underscore + 12 digits for MAC
  ASSERT_EQ(pid_mac.size(), std::to_string(getpid()).size() + 1 + 12);
}
