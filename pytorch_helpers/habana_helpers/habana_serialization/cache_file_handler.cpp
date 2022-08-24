/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "cache_file_handler.h"
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <filesystem>
#include "habana_helpers/logging.h"

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

CacheFileHandler::CacheFileHandler() : curFolderSize{0} {
  maxFolderSize = GET_ENV_FLAG_NEW(PT_CACHE_FOLDER_SIZE_MB);
  maxFolderSize = maxFolderSize * 1024 * 1024;

  const char* s_id = getenv("ID") ? getenv("ID") : "0";
  id = std::atoi(s_id);

  const char* s_rank = getenv("RANK")
      ? getenv("RANK")
      : getenv("OMPI_COMM_WORLD_RANK") ? getenv("OMPI_COMM_WORLD_RANK") : "0";
  rank = std::atoi(s_rank);
}

void CacheFileHandler::init(std::string path) {
  cache_path = std::move(path);

  std::filesystem::path dir_path{cache_path};
  HABANA_ASSERT(std::filesystem::exists(dir_path));
  if (id == 0) {
    auto de = std::filesystem::directory_iterator{dir_path};
    while (de != std::filesystem::end(de)) {
      PT_HABHELPER_DEBUG(
          CACHEFILE_LOG,
          "Cleaning: ",
          de->path(),
          ", Rank: ",
          std::dec,
          getRank());
      std::filesystem::remove(de->path());
      de++;
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

  size = lseek(fd, (size_t)0, SEEK_END);
  lseek(fd, 0, SEEK_SET);

  return true;
}

void CacheFileHandler::addFileInfo(const std::string& cache_id) {
  // Get filename, extract real size, and add
  std::filesystem::path rcpeFile{recipe_file_path(cache_path, cache_id)};
  std::filesystem::path metaFile{metadata_file_path(cache_path, cache_id)};
  HABANA_ASSERT(
      std::filesystem::exists(rcpeFile) && std::filesystem::exists(metaFile));
  std::filesystem::directory_entry de1{rcpeFile};
  std::filesystem::directory_entry de2{metaFile};

  uint64_t size = de1.file_size() + de2.file_size();
  PT_HABHELPER_DEBUG(
      CACHEFILE_LOG,
      "Adding: ",
      cache_id,
      ", Size: ",
      std::dec,
      size,
      ", Rank: ",
      getRank());

  std::lock_guard<std::mutex> lg(mtx);
  curFolderSize += size;
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

void BasicCacheFileHandler::checkAndDelete() {
  std::filesystem::path dir_path{getCachePath()};
  if (!std::filesystem::exists(dir_path))
    return;

  auto de = std::filesystem::directory_iterator{dir_path};
  while ((de != std::filesystem::end(de)) &&
         (curFolderSize > getMaxFolderSize())) {
    std::string cache_id;

    std::string fname = de->path();
    std::string subName = fname.substr(0, fname.rfind("."));

    std::filesystem::path p1{subName + RECIPE_SUFFIX};
    std::filesystem::path p2{subName + METADATA_SUFFIX};

    if (!std::filesystem::exists(p1) || !std::filesystem::exists(p2)) {
      de++;
      continue;

    } else {
      int fd = fileOpen(p2.c_str(), O_RDONLY);
      if (fd < 0 || !fileLock(fd, false)) {
        de++;
        continue;
      }

      // Not closing file because we want to hold the lock
    }

    std::filesystem::directory_entry de1{p1};
    std::filesystem::directory_entry de2{p2};

    uint64_t size = de1.file_size() + de2.file_size();
    PT_HABHELPER_DEBUG(
        CACHEFILE_LOG,
        "Deleting: ",
        subName,
        ", Size: ",
        std::dec,
        size,
        ", Rank: ",
        getRank());

    std::filesystem::remove(p1);
    std::filesystem::remove(p2);

    curFolderSize -= size;

    de++;
  }
}

} // namespace serialization
