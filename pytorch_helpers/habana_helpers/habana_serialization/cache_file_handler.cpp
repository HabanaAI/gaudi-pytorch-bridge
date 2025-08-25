/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "cache_file_handler.h"
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include "recipe_cache_config.h"

#include "habana_helpers/logging.h"
#include "habana_serialization/cache_version.h"

namespace serialization {
namespace {
#define CACHEFILE_LOG "[CACHEFILE] "
} // namespace

std::string recipe_file_path(
    std::string const& path,
    const std::string& cache_id) {
  return path + "/" + cache_id + RECIPE_SUFFIX;
}

std::string metadata_file_path(
    std::string const& path,
    const std::string& cache_id) {
  return path + "/" + cache_id + METADATA_SUFFIX;
}

std::string insert_temp_prefix_filename(std::string const& path) {
  fs::path file_path{path};
  file_path.replace_filename(
      file_path.filename().string().insert(0, TEMP_FILE_PREFIX));
  return file_path.string();
}

std::string append_unique_node_id(std::string const& path) {
  return path + "_" + CacheVersion::combined_pid_mac_addr();
}

CacheFileHandler::CacheFileHandler(const RecipeCacheConfig& recipe_cache_config)
    : maxFolderSize(recipe_cache_config.cache_dir_max_size_mb()) {
  maxFolderSize = maxFolderSize * 1024 * 1024;

  const char* s_local_rank = std::getenv("LOCAL_RANK");
  if (s_local_rank) {
    try {
      local_rank = std::stoi(s_local_rank);
    } catch (const std::invalid_argument& e) {
      local_rank = 0;
    }
  } else {
    local_rank = 0;
  }

  const char* s_rank = std::getenv("RANK");
  if (s_rank) {
    try {
      rank = std::stoi(s_rank);
    } catch (const std::invalid_argument& e) {
      rank = 0;
    }
  } else {
    rank = 0;
  }

  cache_path = recipe_cache_config.path();
  fs::path dir_path{cache_path};
  HABANA_ASSERT(fs::exists(dir_path), "Recipe cache path is expected");
  if (recipe_cache_config.delete_on_init()) {
    if (local_rank == 0) {
      try {
        auto de = fs::directory_iterator{dir_path};
        while (de != fs::end(de)) {
          PT_HABHELPER_DEBUG(
              CACHEFILE_LOG,
              "Cleaning: ",
              Logger::_str_wrapper(de->path()),
              ", Rank: ",
              getRank());
          fs::remove(de->path());
          de++;
        }
      } catch (fs::filesystem_error& err) {
        PT_HABHELPER_DEBUG(
            CACHEFILE_LOG,
            "Exception in cache removal on init, Please delete manually: ",
            err.what(),
            ", Rank: ",
            getRank());
      }
    }
  }
}

int CacheFileHandler::fileOpen(const std::string& fname, int flags) {
  return open(fname.c_str(), flags, S_IRWXU | S_IRWXG | S_IRWXO);
}

int CacheFileHandler::fileClose(int fd) {
  // Function that closes the file, effectively removing the flock on it
  return close(fd);
}

int CacheFileHandler::fileUnLock(int fd) {
  return flock(fd, LOCK_UN);
}

bool CacheFileHandler::fileLock(int fd, bool block) {
  auto flags = LOCK_EX | (block ? 0 : LOCK_NB);

  auto retVal = flock(fd, flags);
  if (retVal == -1) {
    return false;
  }

  return true;
}

bool CacheFileHandler::fileLock(int fd, bool block, size_t& size) {
  if (!fileLock(fd, block))
    return false;

  size = static_cast<size_t>(lseek(fd, 0, SEEK_END));
  lseek(fd, 0, SEEK_SET);

  return true;
}

void CacheFileHandler::addFileInfo(
    const std::string& recipe_file_path,
    const std::string& metadata_file_path) {
  // Get filename, extract real size, and add
  HABANA_ASSERT(fs::exists(recipe_file_path) && fs::exists(metadata_file_path));

#if !defined __GNUC__ || __GNUC__ >= 8
  fs::directory_entry de1{recipe_file_path};
  fs::directory_entry de2{metadata_file_path};
  uint64_t size = de1.file_size() + de2.file_size();
#else
  uint64_t size =
      fs::file_size(recipe_file_path) + fs::file_size(metadata_file_path);
#endif
  PT_HABHELPER_DEBUG(
      CACHEFILE_LOG,
      "Adding: ",
      recipe_file_path,
      " and ",
      metadata_file_path,
      ", Size: ",
      size,
      ", Rank: ",
      getRank());

  std::lock_guard<std::mutex> lg(mtx);
  checkAndDelete();
}

int CacheFileHandler::openAndLockFile(
    const std::string& fname,
    int flags,
    bool block,
    size_t& size) {
  int fd = CacheFileHandler::fileOpen(fname, flags);
  if (fd < 0)
    return fd;

  if (!CacheFileHandler::fileLock(fd, block, size)) {
    CacheFileHandler::fileClose(fd);
    return -1;
  }

  return fd;
}

} // namespace serialization
