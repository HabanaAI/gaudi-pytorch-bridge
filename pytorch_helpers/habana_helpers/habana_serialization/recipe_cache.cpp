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
#include "recipe_cache.h"
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
#include <sstream>
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
// it updates passed stringstream (metadata) and optionally returns
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

bool rename_file(const std::string& old_path, const std::string& new_path) {
  std::error_code err_code;
  fs::rename(old_path, new_path, err_code);
  if (err_code) {
    PT_HABHELPER_ERROR(
        "Cannot rename file ",
        old_path,
        " to ",
        new_path,
        ". Error message :",
        err_code.message());
    return false;
  }
  return true;
}

} // namespace

namespace serialization {

RecipeCache::RecipeCache(const RecipeCacheConfig& recipe_cache_config)
    : cache_path_{recipe_cache_config.path()},
      cf_handler_{nullptr},
      cache_on_nfs_{recipe_cache_config.cache_on_nfs()} {
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
  cache_thread_ = std::make_unique<habana_helpers::JobThread>();
}

RecipeCache::~RecipeCache() {
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

void RecipeCache::lockfree_store_task(
    const std::string& cache_id,
    std::shared_ptr<synapse_helpers::graph::recipe_handle> recipeHandle,
    const std::string& metadata) {
  PT_HABHELPER_DEBUG("Serializing recipe and metadata for cache_id ", cache_id);

  auto recipe_path =
      append_unique_node_id(recipe_file_path(cache_path_, cache_id));
  auto metadata_path =
      append_unique_node_id(metadata_file_path(cache_path_, cache_id));
  auto metadata_path_compiling = insert_temp_prefix_filename(metadata_path);

  auto final_metadata_path = metadata_path;

  if (file_exists(metadata_path_compiling)) {
    metadata_path = metadata_path_compiling;
  } else {
    PT_HABHELPER_DEBUG(
        "Missing metadata for compilation (",
        metadata_path_compiling,
        "). Writing directly to metadata file (",
        metadata_path,
        ").");
  }

  if (recipeHandle && recipeHandle->syn_recipe_handle_ != nullptr) {
    auto serialize_status = synRecipeSerialize(
        recipeHandle->syn_recipe_handle_, recipe_path.c_str());
    if (serialize_status != synSuccess) {
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
      PT_HABHELPER_WARN(
          "Failed to separately open metadata file(",
          metadata_path,
          ") for writing. Err: ",
          err_str,
          ". Removing recipe file as well: ",
          recipe_path);
      fs::remove(recipe_path);
      return;
    }

    metadata_file << metadata;
    metadata_file.close();

    cf_handler_->addFileInfo(recipe_path, metadata_path);

    PT_HABHELPER_DEBUG("Serialization successful for cache_id ", cache_id);

    if (file_exists(metadata_path_compiling)) {
      if (!rename_file(metadata_path_compiling, final_metadata_path)) {
        PT_HABHELPER_DEBUG(
            "Failed to rename temp metadata file ",
            metadata_path_compiling,
            " to final metadata file ",
            final_metadata_path,
            "Removing cache entry entirely, as it's useless.");
        fs::remove(recipe_path);
        fs::remove(metadata_path_compiling);
      }
    }
  } else {
    if (!recipeHandle || recipeHandle->syn_recipe_handle_ == nullptr) {
      PT_HABHELPER_DEBUG("Empty recipe was provided for cache_id ", cache_id);
    }
    PT_HABHELPER_DEBUG("Nothing to serialize.");
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
          metadata_path,
          ") for writing. Err: ",
          err_str);
      return;
    }

    metadata_file << metadata;
    metadata_file.close();

    cf_handler_->addFileInfo(recipe_path, metadata_path);

    PT_HABHELPER_DEBUG("Serialization successful for cache_id ", cache_id);
    cf_handler_->fileUnLock(fd);
    cf_handler_->fileClose(fd);
  } else {
    if (size != 0) {
      PT_HABHELPER_DEBUG(
          "Found non-empty cache entry on disk for cache_id ", cache_id);
    }
    if (!recipeHandle || recipeHandle->syn_recipe_handle_ == nullptr) {
      PT_HABHELPER_DEBUG("Empty recipe was provided for cache_id ", cache_id);
    }
    PT_HABHELPER_DEBUG("Nothing to serialize.");
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
        if (cache_on_nfs_)
          this->lockfree_store_task(cache_id, recipe_handle, metadata);
        else
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

  if (cache_on_nfs_) {
    return lockfree_lookup(cache_id, metadata);
  }

  PT_HABHELPER_DEBUG("Trying to find recipe and metadata for id ", cache_id);

  auto recipe_path = recipe_file_path(cache_path_, cache_id);
  auto metadata_path = metadata_file_path(cache_path_, cache_id);

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

absl::optional<synRecipeHandle> RecipeCache::lockfree_lookup(
    std::string cache_id,
    std::ostream& metadata) {
  if (!is_cache_valid_)
    return {};

  PT_HABHELPER_DEBUG("Trying to find recipe and metadata for id ", cache_id);

  auto recipe_path = recipe_file_path(cache_path_, cache_id);
  auto metadata_path = metadata_file_path(cache_path_, cache_id);
  auto metadata_path_compiling = insert_temp_prefix_filename(metadata_path);

  bool temp_found = false;
  std::string temp_path;
  bool final_found = false;
  std::string final_path;

  // utility func to search for temp cache file and final cache file
  auto search_temp_final = [&]() {
    // all cache entries are suffixed with append_unique_node_id()
    // this search ignores suffixes and any tries to match only cache_id
    for (const auto& entry : fs::directory_iterator(cache_path_)) {
      // check if file matches and is final (not marked for compiling)
      if (entry.path().string().find(metadata_path) != std::string::npos &&
          entry.path().string().find(metadata_path_compiling) ==
              std::string::npos) {
        final_found = true;
        temp_found = false; // temp files irrelevant, when final found
        final_path = entry.path();
        PT_HABHELPER_DEBUG(
            "Found final metadata file: ", final_path, ". Lookup successful.");
        // just in case of intermediate state of creating/moving final vs temp
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // no need to browse the folder any further
        break;
      }
      // check if file matches and is marked for compiling
      // only firstly encoutered temp file is recorded here
      if (!temp_found && entry.path().string().find(metadata_path_compiling)
          != std::string::npos) {
        temp_found = true;
        temp_path = entry.path();
      }
    }
  };

  search_temp_final();
  if (temp_found) {
    PT_HABHELPER_DEBUG(
        "Found temp metadata file: ",
        temp_path,
        ". Other worker is compiling. Waiting for finished compilation on other worker.");
    // In general recipes should not be compiled for more than couple of
    // minutes. However, it's possible, that found temp file is incorrect or
    // leftover from previous run, so in order to not wait indefinitely for
    // nothing, we timeout and check for any final file again.
    auto timeout_counter = GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_NFS_TIMEOUT_S);

    do {
      search_temp_final();
      if (final_found) {
        break; // early break if other final file was created in the meantime
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } while (timeout_counter--);

    // check again after wait (only final matters now)
    if (!final_found) {
      search_temp_final();
    }
  }

  auto create_empty_file = [this](const std::string& path) -> void {
    int fd = cf_handler_->fileOpen(path.c_str(), O_RDONLY | O_CREAT);
    if (fd < 0) {
      PT_HABHELPER_WARN(
          "Can't create empty file: ", path, ", errno: ", strerror(errno));
      return;
    }
    cf_handler_->fileClose(fd);
  };

  if (final_found) {
    PT_HABHELPER_DEBUG(
        "Final metadata found: ",
        final_path,
        ". Proceeding with deserialization..");
    // final_path contains appended unique node_id of node that compiled
    // the recipe. Extract the suffix and append it to recipe_path
    auto suffix = final_path.substr(
        metadata_path.size(), final_path.size() - metadata_path.size());
    metadata_path = final_path;
    recipe_path += suffix;

    PT_HABHELPER_DEBUG(
        "Retrieving meta and recipe: ", metadata_path, ", ", recipe_path);

    int fd = cf_handler_->fileOpen(metadata_path.c_str(), O_RDONLY);
    if (fd >= 0) {
      PT_HABHELPER_DEBUG("Deserializing cache entry for id ", cache_id);
      auto recipe = get_recipe_handle(metadata_path, metadata, recipe_path);
      cf_handler_->fileClose(fd);
      if (!recipe.has_value()) {
        PT_HABHELPER_DEBUG(
            " No recipe found. Creating temp metadata file for compilation. This node is going to compile.");
        create_empty_file(append_unique_node_id(metadata_path_compiling));
      }
      return recipe;
    } else {
      PT_HABHELPER_DEBUG(
          "Can't read metadata file ",
          metadata_path,
          ", errno: ",
          strerror(errno),
          ". Removing..");
      fs::remove(metadata_path);
      PT_HABHELPER_DEBUG(
          "Creating temp metadata file for compilation. This node is going to compile.");
      create_empty_file(append_unique_node_id(metadata_path_compiling));
    }
  } else {
    PT_HABHELPER_DEBUG(
        " Final metadata not found. Creating temp metadata file for compilation. This node is going to compile.");
    create_empty_file(append_unique_node_id(metadata_path_compiling));
  }

  return {};
}

} // namespace serialization
