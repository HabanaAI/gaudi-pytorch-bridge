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
#include <filesystem>
#include "pytorch_helpers/habana_helpers/habana_serialization/include/habana_serialization/recipe_cache.h"

using namespace serialization;

static const std::string kTestCacheDir = "/tmp/recipe_cache_test/";

class RecipeCacheDiskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fs::remove_all(kTestCacheDir);
    // Do not create directory here; RecipeCache should create it on demand.
    // make sure the environment variable is not set
    UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  }
  void TearDown() override {
    fs::remove_all(kTestCacheDir);
    // clear env var after test
    UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  }

  void PrepareRecipeCacheConfig(
      const std::string& path = kTestCacheDir,
      bool delete_on_init = false,
      unsigned int max_size_mb = 1024,
      bool cache_on_nfs = false) {
    std::string value = path + "," + (delete_on_init ? "true" : "false") + "," +
        std::to_string(max_size_mb) + "," + (cache_on_nfs ? "true" : "false");
    SET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG, value.c_str(), 1);
  }
};

TEST_F(RecipeCacheDiskTest, RecipeCacheConfigDefaultConfig) {
  RecipeCacheConfig config;
  EXPECT_EQ(config.path(), "");
  EXPECT_FALSE(config.delete_on_init());
  EXPECT_FALSE(config.cache_on_nfs());
  EXPECT_EQ(config.cache_dir_max_size_mb(), 1024);
}

TEST_F(RecipeCacheDiskTest, RecipeCacheConfigCustomConfigNoNFS) {
  PrepareRecipeCacheConfig(kTestCacheDir, true, 2048, false);
  RecipeCacheConfig config;
  EXPECT_EQ(config.path(), kTestCacheDir);
  EXPECT_EQ(config.delete_on_init(), true);
  EXPECT_EQ(config.cache_dir_max_size_mb(), 2048);
  EXPECT_EQ(config.cache_on_nfs(), false);
}

TEST_F(RecipeCacheDiskTest, RecipeCacheConfigCustomConfigNFS) {
  PrepareRecipeCacheConfig(kTestCacheDir, true, 2048, true);
  RecipeCacheConfig config;
  EXPECT_EQ(config.path(), kTestCacheDir);
  // when NFS enabled, delete is disabled
  EXPECT_EQ(config.delete_on_init(), false);
  // when NFS is enabled, no retention on disk cache is possible
  EXPECT_EQ(config.cache_dir_max_size_mb(), 0);
  EXPECT_EQ(config.cache_on_nfs(), true);
}

TEST_F(RecipeCacheDiskTest, CreatesDirectoryOnDemand) {
  PrepareRecipeCacheConfig(kTestCacheDir, true, 2048, false);
  RecipeCacheConfig config;
  ASSERT_FALSE(fs::exists(kTestCacheDir));
  RecipeCache cache(config);
  EXPECT_TRUE(fs::exists(kTestCacheDir));
}

// TODO: Add tests with actual synRecipe store/lookup

// TODO: Add test with NFS enabled where failed lookup creates temp_* file
