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

#include "const_section.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#include <string>
#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

#if !defined __GNUC__ || __GNUC__ >= 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace serialization {

ConstSectionFileHandler::ConstSectionFileHandler() {
  const char* s_rank = getenv("RANK") ? getenv("RANK") : "0";
  m_rank = std::atoi(s_rank);
}

void ConstSectionFileHandler::internal_mkdir(std::string path) {
  // no checking of retval, the dir is queried below regardless
  PT_CONST_SECTION_DEBUG("Creating const section cache dir: ", path);
  mkdir(path.c_str(), S_IRWXU | S_IRWXG);
  struct stat info {};
  if (stat(path.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
    PT_CONST_SECTION_FATAL("Cannot create const cache directory: ", path);
  } else {
    PT_CONST_SECTION_DEBUG("Cache directory(", path, ") set up properly.");
  }
}

void ConstSectionFileHandler::init(std::string path) {
  if (path == "") {
    return;
  }
  internal_mkdir(path);
  cache_path_ = std::move(path) + "/" + std::to_string(getRank());
  internal_mkdir(cache_path_);

  PT_CONST_SECTION_DEBUG(
      __func__, " Initialing const section path to: ", cache_path_)
  fs::path dir_path{cache_path_};
  HABANA_ASSERT(
      fs::exists(dir_path), "Const section serialize path is expected");
  if (habana_helpers::ShouldClearConstSectionPath()) {
    try {
      auto de = fs::directory_iterator{dir_path};
      while (de != fs::end(de)) {
        PT_CONST_SECTION_DEBUG(
            "Cleaning: ",
            Logger::_str_wrapper(de->path()),
            ", Rank: ",
            getRank());
        fs::remove(de->path());
        de++;
      }
    } catch (fs::filesystem_error& err) {
      PT_CONST_SECTION_FATAL(
          "Exception in const section removal on init, Please delete manually: ",
          err.what(),
          ", Rank: ",
          getRank());
    }
  }
}

std::string ConstSectionDataSerialize::getSerializedFullPath(int const_id) {
  HABANA_ASSERT(habana_helpers::IsConstSectionSerialization());
  return habana_helpers::GetConstSectionSerializationPath() + "/" +
      std::to_string(m_constSectFH->getRank()) + "/" +
      CONST_SECTION_DATA_PREFIX + std::to_string(const_id) +
      CONST_SECTION_DATA_SUFFIX;
}

std::string ConstSectionDataSerialize::getSerializedRecipeFullPath(
    int const_id,
    const size_t key) {
  std::string path = GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH);
  auto full_path = path + "/" + std::to_string(key) + "_" +
      CONST_SECTION_DATA_PREFIX + std::to_string(const_id) +
      CONST_SECTION_DATA_SUFFIX;
  return full_path;
}

bool ConstSectionDataSerialize::fileExists(int const_id) {
  std::lock_guard<std::mutex> lock(m_mtx);
  struct stat buffer;
  bool exists = (stat(getSerializedFullPath(const_id).c_str(), &buffer) == 0);
  if (exists) {
    PT_CONST_SECTION_DEBUG(
        __func__, " file alredy exists: ", getSerializedFullPath(const_id));
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
    int data_size,
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
  if (data) {
    outputFile.write(reinterpret_cast<const char*>(data), data_size);
  }
  outputFile.close();
}

void ConstSectionDataSerialize::compress_and_serialize(
    void* data,
    int data_size,
    std::ofstream& outputFile) {
  z_stream zs;
  memset(&zs, 0, sizeof(zs));
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

  zs.next_in = static_cast<Bytef*>(const_cast<void*>(data));
  zs.avail_in = data_size;

  int ret;
  char outbuffer[data_size];

  do {
    zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
    zs.avail_out = sizeof(outbuffer);

    ret = deflate(&zs, Z_FINISH);

    outputFile.write(outbuffer, zs.total_out - outputFile.tellp());
  } while (ret == Z_OK);

  deflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    throw std::runtime_error("Error while compressing: " + std::to_string(ret));
  }
}

void ConstSectionDataSerialize::serialize(
    void* data,
    int data_size,
    int const_id) {
  std::lock_guard<std::mutex> lock(m_mtx);
  PT_CUSTOM_DEBUG(__func__, ": ", getSerializedFullPath(const_id))
  std::ofstream outputFile(
      getSerializedFullPath(const_id), std::ios::out | std::ios::binary);
  if (!outputFile) {
    PT_CONST_SECTION_FATAL(
        "Cannot open const section file ectory for writing: ",
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
    outputFile.write(reinterpret_cast<const char*>(data), data_size);
  }
  outputFile.close();
  m_isSerialized = true;
}

void ConstSectionDataSerialize::decompress_and_deserialize(
    void* data,
    int data_size,
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
  zs.avail_in = inputFile.tellg();

  zs.next_out = static_cast<Bytef*>(data);
  zs.avail_out = data_size;

  int ret;

  do {
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
    int data_size,
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
  inputFile.read(reinterpret_cast<char*>(data), data_size);
  inputFile.close();
}

void ConstSectionDataSerialize::deserialize(
    void* data,
    int data_size,
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
  std::streampos size = inputFile.tellg();
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
      inputFile.read(reinterpret_cast<char*>(data), data_size);
    }
    inputFile.close();
  }
}

} // namespace serialization