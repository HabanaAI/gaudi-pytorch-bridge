/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "const_section.h"
#include <absl/container/fixed_array.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <zconf.h>
#include <zlib.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/env_flags.h" // NOLINT(misc-include-cleaner)
#include "habana_helpers/logging.h"
#include "recipe_cache_config.h"

namespace serialization {

ConstSectionFileHandler::ConstSectionFileHandler() {
  const char* s_rank = std::getenv("RANK");
  m_rank = (s_rank != nullptr) ? std::atoi(s_rank) : 0;
}

void ConstSectionFileHandler::internal_mkdir(std::string path) {
  // no checking of retval, the dir is queried below regardless
  PT_CONST_SECTION_DEBUG("Creating const section cache dir: ", path);
  mkdir(path.c_str(), S_IRWXU | S_IRWXG);
  struct stat info{};
  if (stat(path.c_str(), &info) != 0 || ((info.st_mode & S_IFDIR) == 0U)) {
    PT_CONST_SECTION_FATAL("Cannot create const cache directory: ", path);
  } else {
    PT_CONST_SECTION_DEBUG("Cache directory(", path, ") set up properly.");
  }
}

void ConstSectionFileHandler::init(std::string path) {
  if (path.empty()) {
    return;
  }
  internal_mkdir(path);
  cache_path_ = std::move(path) + "/" + std::to_string(getRank());
  internal_mkdir(cache_path_);

  PT_CONST_SECTION_DEBUG(
      __func__, " Initialing const section path to: ", cache_path_)
  std::filesystem::path dir_path{cache_path_};
  HABANA_ASSERT(
      std::filesystem::exists(dir_path),
      "Const section serialize path is expected");
  if (habana_helpers::ShouldClearConstSectionPath()) {
    try {
      auto de = std::filesystem::directory_iterator{dir_path};
      while (de != std::filesystem::end(de)) {
        PT_CONST_SECTION_DEBUG(
            "Cleaning: ",
            Logger::_str_wrapper(de->path()),
            ", Rank: ",
            getRank());
        std::filesystem::remove(de->path());
        de++;
      }
    } catch (std::filesystem::filesystem_error& err) {
      PT_CONST_SECTION_FATAL(
          "Exception in const section removal on init, Please delete manually: ",
          err.what(),
          ", Rank: ",
          getRank());
    }
  }
}

namespace {
using namespace std::literals;

constexpr const auto CONST_SECTION_DATA_PREFIX = "const_tensor_"sv;
constexpr const auto CONST_SECTION_DATA_SUFFIX = ".data"sv;
constexpr const auto CONST_SECTION_COMPRESSION_CHUNK_SIZE = 32768;
} // namespace

std::string ConstSectionDataSerialize::getSerializedFullPath(int const_id) {
  HABANA_ASSERT(habana_helpers::IsConstSectionSerialization());
  return habana_helpers::GetConstSectionSerializationPath()
      .append("/"sv)
      .append(std::to_string(m_constSectFH->getRank()))
      .append("/"sv)
      .append(CONST_SECTION_DATA_PREFIX)
      .append(std::to_string(const_id))
      .append(CONST_SECTION_DATA_SUFFIX);
}

std::string ConstSectionDataSerialize::getSerializedRecipeFullPath(
    int const_id,
    const size_t key) {
  static const std::string cache_path = [] {
    std::vector<std::string> split_config =
        RecipeCacheConfig::split_params(GET_ENV_FLAG_NEW(
            PT_HPU_RECIPE_CACHE_CONFIG)); // NOLINT(misc-include-cleaner)
    return (split_config.empty() ? "" : split_config[0]) + "/";
  }();
  std::string result;
  constexpr auto underscore = "_"sv;
  constexpr int size_t_length = 21;
  result.reserve(
      cache_path.size() + size_t_length + underscore.length() +
      CONST_SECTION_DATA_PREFIX.size() + size_t_length +
      CONST_SECTION_DATA_SUFFIX.size());
  return result.append(cache_path)
      .append(std::to_string(key))
      .append(underscore)
      .append(CONST_SECTION_DATA_PREFIX)
      .append(std::to_string(const_id))
      .append(CONST_SECTION_DATA_SUFFIX);
}

bool ConstSectionDataSerialize::fileExists(int const_id) {
  std::lock_guard<std::mutex> lock(m_mtx);
  struct stat buffer;
  bool exists = (stat(getSerializedFullPath(const_id).c_str(), &buffer) == 0);
  if (exists) {
    PT_CONST_SECTION_DEBUG(
        __func__, " file already exists: ", getSerializedFullPath(const_id));
  }
  return exists;
}

ConstSectionDataSerialize::ConstSectionDataSerialize() {
  m_constSectFH = ConstSectionFileHandlerSingleton::getInstance();
}

bool ConstSectionDataSerialize::isSerialized(int const_id) {
  return m_isSerialized || fileExists(const_id);
}

void ConstSectionDataSerialize::serializePerRecipe(
    void* data,
    size_t data_size,
    int const_id,
    const size_t key) {
  PT_CUSTOM_DEBUG(__func__, ": ", getSerializedRecipeFullPath(const_id, key))
  std::ofstream outputFile(
      getSerializedRecipeFullPath(const_id, key),
      std::ios::out | std::ios::binary);
  if (!outputFile) {
    PT_CONST_SECTION_FATAL(
        "Cannot open const section file directory for writing: ",
        getSerializedRecipeFullPath(const_id, key));
    return;
  }

  PT_CONST_SECTION_DEBUG(
      __func__,
      " Dumping tensor recipe data to disk: ",
      getSerializedRecipeFullPath(const_id, key),
      " size: ",
      data_size);

  // if section size is 0, data pointer will be null
  if (data != nullptr) {
    HABANA_ASSERT(
        data_size <= std::numeric_limits<std::streamsize>::max(),
        "Data size exceeds maximum stream size limit for serialization.");
    outputFile.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(data_size));
  }
  outputFile.close();
}

void ConstSectionDataSerialize::compress_and_serialize(
    void* data,
    size_t data_size,
    std::ofstream& outputFile) {
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  // NOLINTNEXTLINE(readability-magic-numbers)
  int window_bits = 15 | 16; /*The base two logarithm of the window size (the
                                size of the history buffer).*/
  int mem_level =
      8; /*The memory usage level, ranging from 1 to 9. A higher value uses more
            memory for optimization. 8 is the default.*/
  if (deflateInit2(
          &zs,
          Z_BEST_COMPRESSION,
          Z_DEFLATED,
          window_bits,
          mem_level,
          Z_DEFAULT_STRATEGY) != Z_OK) {
    throw std::runtime_error("deflateInit2 failed while compressing.");
  }

  zs.next_in = static_cast<Bytef*>(data);
  HABANA_ASSERT(
      data_size <= std::numeric_limits<unsigned int>::max(),
      "Data size exceeds maximum unsigned int limit for compression.");
  zs.avail_in = static_cast<unsigned int>(data_size);

  int ret;
  absl::FixedArray<char> outbuffer(CONST_SECTION_COMPRESSION_CHUNK_SIZE);

  do { // NOLINT(cppcoreguidelines-avoid-do-while)
    zs.next_out = reinterpret_cast<Bytef*>(outbuffer.data());
    static_assert(
        CONST_SECTION_COMPRESSION_CHUNK_SIZE * sizeof(char) <=
            std::numeric_limits<unsigned int>::max(),
        "CONST_SECTION_COMPRESSION_CHUNK_SIZE exceeds maximum unsigned int limit for compression.");
    zs.avail_out = static_cast<unsigned int>(outbuffer.memsize());

    ret = deflate(&zs, Z_FINISH);

    HABANA_ASSERT(
        zs.total_out <= std::numeric_limits<std::streamsize>::max(),
        "Compressed data size exceeds maximum stream size limit.");
    outputFile.write(
        outbuffer.data(), static_cast<long>(zs.total_out) - outputFile.tellp());
  } while (ret == Z_OK);

  deflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    throw std::runtime_error("Error while compressing: " + std::to_string(ret));
  }
}

void ConstSectionDataSerialize::serialize(
    void* data,
    size_t data_size,
    int const_id) {
  std::lock_guard<std::mutex> lock(m_mtx);
  PT_CUSTOM_DEBUG(__func__, ": ", getSerializedFullPath(const_id))
  std::ofstream outputFile(
      getSerializedFullPath(const_id), std::ios::out | std::ios::binary);
  if (!outputFile) {
    PT_CONST_SECTION_FATAL(
        "Cannot open const section file for writing: ",
        getSerializedFullPath(const_id));
    return;
  }

  PT_CONST_SECTION_DEBUG(
      __func__,
      " Dumping tensor host data to disk: ",
      getSerializedFullPath(const_id),
      " size: ",
      data_size);
  if (habana_helpers::IsCompressionEnabled()) {
    compress_and_serialize(data, data_size, outputFile);
  } else {
    HABANA_ASSERT(
        data_size <= std::numeric_limits<std::streamsize>::max(),
        "Data size exceeds maximum stream size limit for serialization.");
    outputFile.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(data_size));
  }
  outputFile.close();
  m_isSerialized = true;
}

void ConstSectionDataSerialize::decompress_and_deserialize(
    void* data,
    size_t data_size,
    std::ifstream& inputFile) {
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  int window_bits = 16;
  if (inflateInit2(&zs, window_bits + MAX_WBITS) != Z_OK) {
    throw std::runtime_error("inflateInit2 failed while decompressing.");
  }
  std::vector<char> compressedData(
      (std::istreambuf_iterator<char>(inputFile)),
      std::istreambuf_iterator<char>());
  zs.next_in =
      reinterpret_cast<Bytef*>(const_cast<char*>(compressedData.data()));
  const auto read_pos = inputFile.tellg();
  HABANA_ASSERT(
      read_pos <= std::numeric_limits<unsigned int>::max(),
      "Input file size exceeds maximum unsigned int limit for decompression.");
  zs.avail_in = static_cast<unsigned int>(inputFile.tellg());

  zs.next_out = static_cast<Bytef*>(data);
  HABANA_ASSERT(
      data_size <= std::numeric_limits<unsigned int>::max(),
      "Data size exceeds maximum unsigned int limit for decompression.");
  zs.avail_out = static_cast<unsigned int>(data_size);

  int ret;

  do { // NOLINT(cppcoreguidelines-avoid-do-while)
    ret = inflate(&zs, Z_NO_FLUSH);
  } while (ret == Z_OK);

  inflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    throw std::runtime_error(
        "Error while decompressing: " + std::to_string(ret));
  }
}

void ConstSectionDataSerialize::deserializePerRecipe(
    void* data,
    size_t data_size,
    int const_id,
    const size_t key) {
  PT_CUSTOM_DEBUG(__func__, ": ", getSerializedRecipeFullPath(const_id, key))
  HABANA_ASSERT(
      data,
      "Got a nullptr for deserialize const section: ",
      getSerializedRecipeFullPath(const_id, key));

  std::ifstream inputFile(
      getSerializedRecipeFullPath(const_id, key),
      std::ios::in | std::ios::binary);
  if (!inputFile) {
    PT_CONST_SECTION_FATAL(
        "Error opening const section file ",
        getSerializedRecipeFullPath(const_id, key));
  }

  PT_CONST_SECTION_DEBUG(
      "Loaded tensor host data from disk: ",
      getSerializedRecipeFullPath(const_id, key),
      " size: ",
      data_size);
  HABANA_ASSERT(
      data_size <= std::numeric_limits<std::streamsize>::max(),
      "Data size exceeds maximum stream size limit for deserialization.");
  inputFile.read(
      reinterpret_cast<char*>(data), static_cast<std::streamsize>(data_size));
  inputFile.close();
}

void ConstSectionDataSerialize::deserialize(
    void* data,
    size_t data_size,
    int const_id) {
  std::lock_guard<std::mutex> lock(m_mtx);
  PT_CUSTOM_DEBUG(__func__, ": ", getSerializedFullPath(const_id))
  if (reinterpret_cast<char*>(data) == nullptr) {
    PT_CONST_SECTION_FATAL(
        "Got a nullptr for deserialize const section: ",
        getSerializedFullPath(const_id));
  }
  std::ifstream inputFile(
      getSerializedFullPath(const_id), std::ios::in | std::ios::binary);
  if (!inputFile) {
    PT_CONST_SECTION_FATAL(
        "Error opening const section file ", getSerializedFullPath(const_id));
  }

  inputFile.seekg(0, std::ios::end);
  auto size = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  if (size == 0) {
    PT_CONST_SECTION_WARN(
        "Const section file is empty: ", getSerializedFullPath(const_id));
    inputFile.close();
  } else {
    HABANA_ASSERT(habana_helpers::IsConstSectionSerialization());
    PT_CONST_SECTION_DEBUG(
        "Loaded tensor host data from disk: ",
        getSerializedFullPath(const_id),
        " size: ",
        data_size);
    if (habana_helpers::IsCompressionEnabled()) {
      decompress_and_deserialize(data, data_size, inputFile);
    } else {
      HABANA_ASSERT(
          data_size <= std::numeric_limits<std::streamsize>::max(),
          "Data size exceeds maximum stream size limit for deserialization.");
      inputFile.read(
          reinterpret_cast<char*>(data),
          static_cast<std::streamsize>(data_size));
    }
    inputFile.close();
  }
}

} // namespace serialization
