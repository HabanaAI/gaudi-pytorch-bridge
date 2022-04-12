/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "recipe_cache.h"
#include <c10/util/Exception.h>
#include <errno.h>
#include <fcntl.h>
#include <synapse_api.h>
#include <synapse_common_types.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <memory>
#include "habana_helpers/logging.h"

constexpr const char* RECIPE_SUFFIX = ".recipe";
constexpr const char* METADATA_SUFFIX = ".metadata";

namespace {

std::string recipe_file_path(std::string const& path, std::string cache_id) {
  return path + "/" + cache_id + RECIPE_SUFFIX;
}

std::string metadata_file_path(std::string const& path, std::string cache_id) {
  return path + "/" + cache_id + METADATA_SUFFIX;
}

bool file_exists(std::string const& file) {
  struct stat buffer; // NOLINT
  return stat(file.c_str(), &buffer) == 0;
}

// Function that closes the file, effectively removing the flock on it
void unlock_file(std::string const& f_path, int fd) {
  PT_HABHELPER_DEBUG("Unlocking and closing ", f_path, " ...");
  // Sanity check. File must always exist for this function.
  if (!file_exists(f_path)) {
    PT_HABHELPER_FATAL(
        "There is no file ( ", f_path, ") to handle. Something went wrong...");
    TORCH_CHECK(false, "recipe save to disk failed");
  }
  auto retVal = close(fd);
  if (retVal) {
    PT_HABHELPER_FATAL(
        "Could not close the file ", f_path, ", err: ", strerror(errno));
    TORCH_CHECK(false, "recipe save to disk failed");
  }

  PT_HABHELPER_TRACE(f_path, " unlocked and closed.");
}

// utility function to retrieve valid recipe&metadata
// it updates passed stringstream (metadata) and optinally returns
// synRecipeHandle, if exists
absl::optional<synRecipeHandle> get_recipe_handle(
    const std::string& metadata_path,
    std::ostream& metadata,
    const std::string& recipe_path) {
  {
    std::ifstream metadata_file(metadata_path.c_str(), std::ifstream::binary);
    if (!metadata_file) {
      PT_HABHELPER_TRACE("Failed to open metadata file ", metadata_path);
      return {};
    }
    metadata << metadata_file.rdbuf();

    if (!metadata_file) {
      PT_HABHELPER_TRACE("Failed to open metadata file ", metadata_path);
      return {};
    }
  }

  // conditionally deserialize recipe file. It can be missing, so we return
  // valid optional - nullptr.
  if (file_exists(recipe_path)) {
    synRecipeHandle recipeHandle;
    auto status = synRecipeDeSerialize(&recipeHandle, recipe_path.c_str());
    if (status != synSuccess) {
      PT_HABHELPER_TRACE("Failed to odeserialize recipe with error:  ", status);
      return {};
    }
    PT_HABHELPER_TRACE("Found cache entry with recipe: ", recipe_path);
    return recipeHandle;
  } else {
    PT_HABHELPER_TRACE(
        "Found cache entry without recipe: ",
        recipe_path,
        "- probably empty recipe cached.");
    return nullptr;
  }
  return {};
}

} // namespace

namespace serialization {

RecipeCache::RecipeCache(std::string cache_path)
    : mut_{},
      cond_var_{},
      cache_path_{std::move(cache_path)},
      is_cache_valid_{false},
      inter_host_cache_{nullptr} {
  if (GET_ENV_FLAG_NEW(PT_ENABLE_INTER_HOST_CACHING)) {
    inter_host_cache_ = std::make_unique<InterHostCache>(cache_path_);
    inter_host_cache_->init();
  }

  // no checking of retval, the dir is queried below regardless
  mkdir(cache_path_.c_str(), S_IRWXU | S_IRWXG);
  struct stat info {};
  if (stat(cache_path_.c_str(), &info) != 0 ||
      !(info.st_mode & S_IFDIR)) { // NOLINT(hicpp-signed-bitwise))
    PT_HABHELPER_DEBUG("Cannot create cache directory ", cache_path_);
  } else {
    PT_HABHELPER_DEBUG("Cache directory(", cache_path_, ") set up properly.");
    is_cache_valid_ = true;
  }
}

RecipeCache::~RecipeCache() = default;

void RecipeCache::store(
    std::string cache_id,
    std::shared_ptr<synapse_helpers::graph::recipe_handle> const& recipeHandle,
    std::stringstream&& metadata) {
  if (!is_cache_valid_)
    return;

  PT_HABHELPER_DEBUG("Serializing recipe and metadata for cache_id ", cache_id);

  auto recipe_path = recipe_file_path(cache_path_, cache_id);
  auto metadata_path = metadata_file_path(cache_path_, cache_id);
  int meta_fd_to_unlock = pop_meta_fd(metadata_path);

  if (recipeHandle && recipeHandle->syn_recipe_handle_ != nullptr) {
    auto status = synRecipeSerialize(
        recipeHandle->syn_recipe_handle_, recipe_path.c_str());
    if (status != synSuccess) {
      unlock_file(metadata_path, meta_fd_to_unlock);
      PT_HABHELPER_WARN(
          "Failed to serialized recipe(", recipe_path, "). Err: ", status);
      return;
    }
  }

  std::ofstream metadata_file(metadata_path.c_str(), std::ofstream::binary);
  if (!metadata_file.is_open()) {
    auto err_str = strerror(errno);
    unlock_file(metadata_path, meta_fd_to_unlock);
    PT_HABHELPER_WARN(
        "Failed to separately open metadata file(",
        recipe_path,
        ") for writing. Err: ",
        err_str);
    return;
  }
  metadata_file << metadata.rdbuf();
  metadata_file.close();

  PT_HABHELPER_DEBUG("Serialization successful for cache_id ", cache_id);
  unlock_file(metadata_path, meta_fd_to_unlock);

  if (inter_host_cache_) {
    inter_host_cache_->send_file(cache_id);
  }
}

absl::optional<synRecipeHandle> RecipeCache::lookup(
    std::string cache_id,
    std::ostream& metadata) {
  if (!is_cache_valid_)
    return {};

  PT_HABHELPER_DEBUG("Trying to find recipe and metadata for id ", cache_id);

  auto recipe_path = recipe_file_path(cache_path_, cache_id);
  auto metadata_path = metadata_file_path(cache_path_, cache_id);

  if (inter_host_cache_) {
    inter_host_cache_->recv_file(cache_id);
  }

  auto try_lock_and_read = [&,
                            this](int fd) -> absl::optional<synRecipeHandle> {
    // VLOG(10) << "Trying to lock exclusively metadata file " << metadata_path;
    auto retVal = flock(fd, LOCK_EX);
    if (retVal == -1)
      PT_HABHELPER_WARN(
          "Error when locking the metadata file ",
          metadata_path,
          ", err: ",
          strerror(errno));

    auto currentPos = lseek(fd, (size_t)0, SEEK_CUR);
    bool isMetaEmpty = lseek(fd, (size_t)0, SEEK_END) == 0;
    lseek(fd, currentPos, SEEK_SET); // seek back to the beginning of file
    if (isMetaEmpty) {
      PT_HABHELPER_DEBUG(
          "Metadata is empty. This process can compile recipe. Saving fd for metadata file ",
          metadata_path);
      std::unique_lock<std::mutex> lock(mut_);
      meta2fd_map_.emplace(metadata_path, fd);
      return {};
    } else {
      PT_HABHELPER_DEBUG(
          "Metadata file ",
          metadata_path,
          " is not empty. Found valid cache entry.");
      unlock_file(metadata_path, fd);
      PT_HABHELPER_DEBUG("Deserializing cache entry for id ", cache_id);
      return get_recipe_handle(metadata_path, metadata, recipe_path);
    }
  };

  PT_HABHELPER_DEBUG(
      "Trying to exclusively create or open metadata file ", metadata_path);
  auto fd = open(
      metadata_path.c_str(), O_CREAT | O_EXCL, S_IRWXU | S_IRWXG | S_IRWXO);
  if (fd >= 0) {
    return try_lock_and_read(fd);
  } else {
    if (errno != EEXIST)
      PT_HABHELPER_FATAL("Unexpected metadata file open error ", errno);
    PT_HABHELPER_TRACE("Trying to open existing metadata file ", metadata_path);
    fd = open(metadata_path.c_str(), O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
    if (fd >= 0) {
      return try_lock_and_read(fd);
    } else {
      PT_HABHELPER_FATAL(
          "Could not open existing metadata file ",
          metadata_path,
          ", err: ",
          strerror(errno));
    }
  }
  return {};
}

int RecipeCache::pop_meta_fd(const std::string& metadata_file) {
  std::unique_lock<std::mutex> lock(mut_);
  auto it = meta2fd_map_.find(metadata_file);
  HABANA_ASSERT(it != meta2fd_map_.end());
  int fd = it->second;
  meta2fd_map_.erase(it);
  return fd;
}

} // namespace serialization
