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

#include <stddef.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include "recipe_cache_config.h"

#if !defined __GNUC__ || __GNUC__ >= 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace serialization {

constexpr const char* RECIPE_SUFFIX = ".recipe";
constexpr const char* METADATA_SUFFIX = ".metadata";

std::string recipe_file_path(
    std::string const& path,
    const std::string& cache_id);

std::string metadata_file_path(
    std::string const& path,
    const std::string& cache_id);

class CacheFileHandler {
  // This is an Abstract class

 private:
  int local_rank, rank;
  // Updated once during Construction using env: PT_RECIPE_CACHE_PATH
  std::string cache_path;
  // Updated once during Construction using env: PT_CACHE_FOLDER_SIZE_MB
  uint64_t maxFolderSize;

  std::mutex mtx;

 protected:
  std::optional<uint64_t> getMaxFolderSize() {
    // Due to the fact that eviction is performed after recipe storing, there is
    // a chance to exceed the disk cache size defined by user via
    // PT_CACHE_FOLDER_SIZE_MB. In order to avoid such a scenario the max size
    // is limited to the 99% of defined threshold.
    constexpr double threshold_prescaler = 0.99;

    // If set to 0 then recipe cache eviction is disabled.
    if (maxFolderSize == 0) {
      return std::nullopt;
    }

    return threshold_prescaler * static_cast<double>(maxFolderSize);
  }

  // Child classes can view 'cache_path', but can not change it
  const std::string& getCachePath() {
    return cache_path;
  }

  int getRank() {
    return rank;
  }

  // This function implements the eviction policy
  virtual void checkAndDelete() = 0;

 public:
  CacheFileHandler(const RecipeCacheConfig& recipe_cache_config);
  virtual ~CacheFileHandler() = default;

  static int fileOpen(const std::string& fname, int flags);
  static int fileClose(int fd);
  static bool fileLock(int fd, bool block);
  static int fileUnLock(int fd);
  // Lock file and get size
  static bool fileLock(int fd, bool block, size_t& size);
  // Open, Lock, and get Size
  static int openAndLockFile(
      const std::string& fname,
      int flags,
      bool block,
      size_t& size);
  // Add the size of new file and delete something if required
  void addFileInfo(const std::string& cache_id);
};

} // namespace serialization
