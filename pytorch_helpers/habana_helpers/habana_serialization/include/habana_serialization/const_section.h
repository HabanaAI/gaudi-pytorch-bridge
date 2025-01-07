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
