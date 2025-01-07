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
#include <future>
#include <memory>
#include <sstream>
#include <utility>
#include "base_cache_file_handler.h"
#include "habana_helpers/logging.h"

#if !defined __GNUC__ || __GNUC__ >= 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace {

bool file_exists(std::string const& file) {
  struct stat buffer; // NOLINT
  return stat(file.c_str(), &buffer) == 0;
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
      PT_HABHELPER_WARN("Failed to open metadata file ", metadata_path);
      return {};
    }
    metadata << metadata_file.rdbuf();

    if (!metadata_file) {
      PT_HABHELPER_WARN("Failed to open metadata file ", metadata_path);
      return {};
    }
  }

  // conditionally deserialize recipe file. It can be missing, so we return
  // valid optional - nullptr.
  if (file_exists(recipe_path)) {
    synRecipeHandle recipeHandle;
    auto status = synRecipeDeSerialize(&recipeHandle, recipe_path.c_str());
    if (status != synSuccess) {
      PT_HABHELPER_WARN(
          Logger::formatStatusMsg(status), "Failed to deserialize recipe");
      return {};
    }
    PT_HABHELPER_DEBUG("Found cache entry with recipe: ", recipe_path);
    return recipeHandle;
  } else {
    PT_HABHELPER_DEBUG(
        "Found cache entry without recipe: ",
        recipe_path,
        "- probably empty recipe cached.");
    return nullptr;
  }
  return {};
}

} // namespace

namespace serialization {

RecipeCache::RecipeCache(const RecipeCacheConfig& recipe_cache_config)
    : cache_path_{recipe_cache_config.path()},
      is_cache_valid_{false},
      inter_host_cache_{nullptr},
      cf_handler_{nullptr} {
  std::error_code err_code;
  bool newly_created = fs::create_directories(cache_path_, err_code);

  if (!err_code) {
    if (newly_created) {
#if !defined __GNUC__ || __GNUC__ >= 8
      fs::permissions(
          cache_path_,
          fs::perms::owner_all | fs::perms::group_all,
          fs::perm_options::add);
#else
      fs::permissions(
          cache_path_,
          fs::perms::add_perms | fs::perms::owner_all | fs::perms::group_all);
#endif
    }

    PT_HABHELPER_INFO("Cache directory(", cache_path_, ") set up properly.");
    is_cache_valid_ = true;
  } else {
    PT_HABHELPER_FATAL(
        "Cannot create cache directory(",
        cache_path_,
        "). Error message :",
        err_code.message());
  }

  cf_handler_ = std::make_unique<BaseCacheFileHandler>(recipe_cache_config);

  if (GET_ENV_FLAG_NEW(PT_ENABLE_INTER_HOST_CACHING)) {
    inter_host_cache_ =
        std::make_unique<InterHostCache>(cache_path_, cf_handler_);
    inter_host_cache_->init();
  }

  cache_thread_ = std::make_unique<habana_helpers::JobThread>();
}

RecipeCache::~RecipeCache() {
  // ensure that interhost sync ended
  if (interhost_send_thread_.valid()) {
    interhost_send_thread_.get();
  }

  cache_thread_ = nullptr;
}

void RecipeCache::flush() {
  // wait until cache thread finished its job
  if (cache_thread_) {
    while (cache_thread_->jobCounter() > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void RecipeCache::store_task(
    const std::string& cache_id,
    std::shared_ptr<synapse_helpers::graph::recipe_handle> recipeHandle,
    const std::string& metadata) {
  PT_HABHELPER_DEBUG("Serializing recipe and metadata for cache_id ", cache_id);

  auto recipe_path = recipe_file_path(cache_path_, cache_id);
  auto metadata_path = metadata_file_path(cache_path_, cache_id);

  size_t size;
  int fd = cf_handler_->fileOpen(metadata_path, O_RDWR | O_CREAT);
  if (fd < 0 && errno == EACCES) {
    PT_HABHELPER_WARN("Cannot open cache directory for writing.");
    return;
  }
  bool locked = cf_handler_->fileLock(fd, true, size);
  if (!locked) {
    PT_HABHELPER_WARN(
        "Error when locking the metadata file ",
        metadata_path,
        ", err: ",
        strerror(errno));
    return;
  }

  if (size == 0 && recipeHandle &&
      recipeHandle->syn_recipe_handle_ != nullptr) {
    auto serialize_status = synRecipeSerialize(
        recipeHandle->syn_recipe_handle_, recipe_path.c_str());
    if (serialize_status != synSuccess) {
      cf_handler_->fileUnLock(fd);
      cf_handler_->fileClose(fd);
      PT_HABHELPER_WARN(
          Logger::formatStatusMsg(serialize_status),
          "Failed to serialized recipe(",
          recipe_path,
          ").");
      return;
    }

    std::ofstream metadata_file(metadata_path.c_str(), std::ofstream::binary);
    if (!metadata_file.is_open()) {
      auto err_str = strerror(errno);
      cf_handler_->fileUnLock(fd);
      cf_handler_->fileClose(fd);
      PT_HABHELPER_WARN(
          "Failed to separately open metadata file(",
          recipe_path,
          ") for writing. Err: ",
          err_str);
      return;
    }

    metadata_file << metadata;
    metadata_file.close();

    cf_handler_->addFileInfo(cache_id);

    PT_HABHELPER_DEBUG("Serialization successful for cache_id ", cache_id);
    cf_handler_->fileUnLock(fd);
    cf_handler_->fileClose(fd);

    if (inter_host_cache_) {
      if (interhost_send_thread_.valid()) {
        interhost_send_thread_.get();
      }

      interhost_send_thread_ = std::async(
          std::launch::async, [&] { inter_host_cache_->send_file(cache_id); });
    }
  } else {
    cf_handler_->fileUnLock(fd);
    cf_handler_->fileClose(fd);
  }
}

void RecipeCache::store(
    std::string cache_id,
    std::shared_ptr<synapse_helpers::graph::recipe_handle> recipe_handle,
    const std::stringstream& metadata) {
  if (!is_cache_valid_)
    return;

  PT_HABHELPER_DEBUG("Adding task for storing cache: ", cache_id);
  cache_thread_->addJob(
      [this, cache_id, recipe_handle, metadata = metadata.str()]() {
        this->store_task(cache_id, recipe_handle, metadata);
        PT_HABHELPER_DEBUG("Store task finished: ", cache_id);
        return true;
      });
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
    size_t size;
    bool locked = cf_handler_->fileLock(fd, true, size);
    if (!locked) {
      PT_HABHELPER_WARN(
          "Error when locking the metadata file ",
          metadata_path,
          ", err: ",
          strerror(errno));
      return {};
    }

    if (size == 0) {
      PT_HABHELPER_WARN("Metadata is empty: ", metadata_path);
      fs::remove(metadata_path);
      cf_handler_->fileUnLock(fd);
      return {};
    } else {
      PT_HABHELPER_DEBUG(
          "Metadata file ",
          metadata_path,
          " is not empty. Found valid cache entry.");
      PT_HABHELPER_DEBUG("Deserializing cache entry for id ", cache_id);
      auto recipe = get_recipe_handle(metadata_path, metadata, recipe_path);
      cf_handler_->fileUnLock(fd);
      return recipe;
    }
  };

  int fd = cf_handler_->fileOpen(metadata_path.c_str(), O_RDONLY);
  if (fd >= 0) {
    auto recipe = try_lock_and_read(fd);
    cf_handler_->fileClose(fd);
    return recipe;
  } else {
    PT_HABHELPER_DEBUG(
        "Can't read metadata file ",
        metadata_path,
        ", errno: ",
        strerror(errno));
  }

  return {};
}

} // namespace serialization
