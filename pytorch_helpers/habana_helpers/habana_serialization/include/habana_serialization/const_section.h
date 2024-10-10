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

#include <sys/stat.h>
#include <mutex>
#include <string>
#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/env_flags.h"
#include "cache_file_handler.h"

namespace serialization {

constexpr const char* CONST_SECTION_DATA_PREFIX = "const_tensor_";
constexpr const char* CONST_SECTION_DATA_SUFFIX = ".data";

class ConstSectionFileHandler {
 public:
  ConstSectionFileHandler();
  virtual ~ConstSectionFileHandler() = default;

  // Must be called if const section path changes
  void init(std::string path);

  // TODO: create serialize file eviction methods uppon size
  virtual void checkAndDelete() {}

  int getRank() {
    return m_rank;
  }

 private:
  void internal_mkdir(std::string path);
  int m_rank;
  std::string cache_path_;
};

class ConstSectionFileHandlerSingleton : public ConstSectionFileHandler {
 private:
  ConstSectionFileHandlerSingleton() {
    init(habana_helpers::GetConstSectionSerializationPath());
  }

  // TODO: create serialize file eviction methods uppon size
  void checkAndDelete() override{};

 public:
  static std::shared_ptr<ConstSectionFileHandlerSingleton> getInstance() {
    static std::shared_ptr<ConstSectionFileHandlerSingleton> csHandler(
        new ConstSectionFileHandlerSingleton);
    return csHandler;
  }
};

class ConstSectionDataSerialize {
 public:
  ConstSectionDataSerialize();
  virtual ~ConstSectionDataSerialize() = default;

  void serialize(void* data, int data_size, int const_id);
  void serializePerRecipe(
      void* data,
      int data_size,
      int const_id,
      const size_t key);
  void deserialize(void* data, int data_size, int const_id);
  void deserializePerRecipe(
      void* data,
      int data_size,
      int const_id,
      const size_t key);
  bool isSerialized(int const_id);

  void compress_and_serialize(
      void* data,
      int data_size,
      std::ofstream& outputFile);
  void decompress_and_deserialize(
      void* data,
      int data_size,
      std::ifstream& inputFile);

  std::string getSerializedFullPath(int const_id);
  std::string getSerializedRecipeFullPath(int const_id, const size_t key);

 private:
  bool fileExists(int const_id);

  std::mutex m_mtx;
  bool m_isSerialized{false};
  std::string m_filePathSuffix;

  std::shared_ptr<ConstSectionFileHandler> m_constSectFH;
};

} // namespace serialization
