/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
#include <gtest/gtest.h>
#include <synapse_api_types.h>
#include <synapse_common_types.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

#include <fstream>
#include "backend/helpers/create_tensor.h"
#include "backend/kernel/hpu_habana_cache.h"
#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "pytorch_helpers/habana_helpers/habana_serialization/include/habana_serialization/deserializers.h"
#include "pytorch_helpers/habana_helpers/habana_serialization/include/habana_serialization/serializers.h"

using namespace std;
using namespace habana;

class HabanaSerializationRecipeTest : public ::testing::Test {
  std::string m_cache_path;
  bool m_cache_overriden = false;

  void SetUp() override {
    m_cache_path = GET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH);
    overrideEmptyCachePathEnv();
  }

  void TearDown() override {
    if (m_cache_overriden) {
      SET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH, "", 1); // set empty path
    }
  }

 private:
  void overrideEmptyCachePathEnv() {
    const std::string dafault_cache_path = "cache_dir";
    if (m_cache_path == "") {
      m_cache_overriden = true;
      m_cache_path = dafault_cache_path;
      SET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH, m_cache_path.c_str(), 1);
    }
  }

 protected:
  const std::string& getCachePath() const {
    return m_cache_path;
  }
};

TEST(HabanaSerializationTest, TensorOptionsTest) {
  c10::optional<at::ScalarType> dtype = c10::ScalarType::Float;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions()
          .dtype(dtype)
          .device(hb_device)
          .requires_grad(false)
          .layout(c10::Layout::Strided)
          .pinned_memory(false)
          .memory_format(c10::MemoryFormat::Contiguous);

  std::stringstream ss;
  serialization::serialize(ss, hb_options);

  at::TensorOptions restored_options;
  serialization::deserialize(ss, restored_options);

  ASSERT_EQ(hb_options.device(), restored_options.device());
  ASSERT_EQ(hb_options.layout(), restored_options.layout());
  ASSERT_EQ(hb_options.dtype(), restored_options.dtype());
  ASSERT_EQ(hb_options.requires_grad(), restored_options.requires_grad());
  ASSERT_EQ(hb_options.pinned_memory(), restored_options.pinned_memory());
  ASSERT_EQ(
      hb_options.memory_format_opt(), restored_options.memory_format_opt());
}

int getFilesCount(const char* dir, const char* ext) {
  int count = 0;
  auto fs_path = fs::path(dir);
  auto dirIter = fs::directory_iterator(fs_path);
  for (const auto& file : dirIter) {
    string file_name = file.path().filename();
    if (file_name.find(ext) != string::npos)
      count++;
  }
  return count;
}

int removeFiles(const char* dir) {
  auto fs_path = fs::path(dir);
  int count = 0;
  auto dirIter = fs::directory_iterator(fs_path);
  for (const auto& file : dirIter) {
    if (fs::remove(file.path())) {
      count++;
    }
  }
  return count;
}

TEST_F(HabanaSerializationRecipeTest, serializeDeserializeRecipeTest1) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE)) {
    GTEST_SKIP();
  }
  RecipeCacheLRU::get_cache().ResetDiskCache();
  // make sure dir is empty.
  if (fs::exists(fs::path(getCachePath()))) {
    auto removedFilesCount = removeFiles(getCachePath().c_str());
    size_t cache_size = 0;
    bool dropped = false;
    do {
      dropped = RecipeCacheLRU::get_cache().drop_lru(cache_size);
    } while (cache_size > 0 && dropped);
    HABANA_ASSERT(RecipeCacheLRU::get_cache().empty());
  }

  torch::Tensor originalRecipe = {};
  torch::Tensor deserializedRecipe = {};
  torch::Tensor recipe_in = {};
  torch::Tensor recipe_wt = {};
  for (int i = 0; i < 5; i++) {
    int recipe_no = i % 4;
    if (recipe_no == 0) {
      RecipeCacheLRU::get_cache().clear();
    }
    auto in =
        torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
    auto wt = torch::randn(
        {4, 5 + recipe_no, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
    if (i == 0) {
      recipe_in = in;
      recipe_wt = wt;
    } else if (i == 4) {
      in = recipe_in;
      wt = recipe_wt;
    }
    auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);

    torch::Tensor result =
        torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
    if (i == 0) {
      originalRecipe = result;
    } else if (i == 4) {
      deserializedRecipe = result;
    }

    habana_lazy::HbLazyTensor::StepMarker({});

    // ensure that disk cache thread stored recipes on disk
    RecipeCacheLRU::get_cache().FlushDiskCache();

    int recipe_files_count = getFilesCount(getCachePath().c_str(), ".recipe");
    if (i < 4) {
      ASSERT_EQ(recipe_files_count, (i + 1));
    } else if (i == 4) {
      // last recipe will be deserialized from disk. number of files will not
      // grow
      ASSERT_EQ(recipe_files_count, i);
    }
  }
  auto res2 = deserializedRecipe.to(torch::kCPU);
  auto res1 = originalRecipe.to(torch::kCPU);
  EXPECT_EQ(allclose(res1, res2), true);
  RecipeCacheLRU::get_cache().DeleteDiskCache();
}

TEST_F(HabanaSerializationRecipeTest, serializeDeserializeRecipeTest2) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE)) {
    GTEST_SKIP();
  }
  RecipeCacheLRU::get_cache().ResetDiskCache();
  // make sure dir is empty.
  if (fs::exists(fs::path(getCachePath()))) {
    auto removedFilesCount = removeFiles(getCachePath().c_str());
    size_t cache_size = 0;
    bool dropped = false;
    do {
      dropped = RecipeCacheLRU::get_cache().drop_lru(cache_size);
    } while (cache_size > 0 && dropped);
    HABANA_ASSERT(RecipeCacheLRU::get_cache().empty());
  }
  torch::Tensor originalRecipe = {};
  torch::Tensor deserializedRecipe = {};

  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({4, 5, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
  auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

  for (int i = 0; i < 2; i++) {
    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);

    torch::Tensor result =
        torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
    habana_lazy::HbLazyTensor::StepMarker({});

    // ensure that disk cache thread stored recipes on disk
    RecipeCacheLRU::get_cache().FlushDiskCache();

    int recipe_files_count = getFilesCount(getCachePath().c_str(), ".recipe");
    if (i == 0) {
      originalRecipe = result;
      size_t one = 1;
      RecipeCacheLRU::get_cache().drop_lru(one);
      HABANA_ASSERT(RecipeCacheLRU::get_cache().empty());
      HABANA_ASSERT(recipe_files_count == 1);
    } else if (i == 1) {
      deserializedRecipe = result;
      HABANA_ASSERT(!RecipeCacheLRU::get_cache().empty());
      HABANA_ASSERT(recipe_files_count == 1);
    }
  }
  auto res2 = deserializedRecipe.to(torch::kCPU);
  auto res1 = originalRecipe.to(torch::kCPU);
  EXPECT_EQ(allclose(res1, res2), true);
  RecipeCacheLRU::get_cache().DeleteDiskCache();
}

TEST(HabanaSerializationTest, CharArrayTest) {
  string testArray("this is a serialization test!");
  std::stringstream ss;
  serialization::serialize(ss, testArray.c_str());

  char* restored_testArray;
  serialization::deserialize(ss, restored_testArray);

  string restored_str = string(restored_testArray);

  ASSERT_EQ(testArray, restored_str);
}

TEST(HabanaSerializationTest, StringTest) {
  string testArray("this is a serialization test!");
  std::stringstream ss;
  serialization::serialize(ss, testArray);

  string restored_testArray("");
  serialization::deserialize(ss, restored_testArray);

  ASSERT_EQ(testArray, restored_testArray);
}