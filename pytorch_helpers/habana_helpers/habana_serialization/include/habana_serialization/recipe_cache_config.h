/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
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