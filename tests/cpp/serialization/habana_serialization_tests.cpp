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
    HPUDeviceContext::recipe_cache().ResetDiskCache();
  }

  void TearDown() override {
    if (m_cache_overriden) {
      SET_ENV_FLAG_NEW(PT_RECIPE_CACHE_PATH, "", 1); // set empty path
    }
    HPUDeviceContext::recipe_cache().DeleteDiskCache();
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
  // make sure dir is empty.
  if (fs::exists(fs::path(getCachePath()))) {
    removeFiles(getCachePath().c_str());
    size_t cache_size = 0;
    bool dropped = false;
    do {
      dropped = HPUDeviceContext::recipe_cache().drop_lru(cache_size);
    } while (cache_size > 0 && dropped);
    HABANA_ASSERT(HPUDeviceContext::recipe_cache().empty());
  }

  torch::Tensor originalRecipe = {};
  torch::Tensor deserializedRecipe = {};
  torch::Tensor recipe_in = {};
  torch::Tensor recipe_wt = {};
  for (int i = 0; i < 5; i++) {
    int recipe_no = i % 4;
    if (recipe_no == 0) {
      HPUDeviceContext::recipe_cache().clear();
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
    HPUDeviceContext::recipe_cache().FlushDiskCache();

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
}

TEST_F(HabanaSerializationRecipeTest, serializeDeserializeRecipeTest2) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_PGM_ENABLE_CACHE)) {
    GTEST_SKIP();
  }
  // make sure dir is empty.
  if (fs::exists(fs::path(getCachePath()))) {
    removeFiles(getCachePath().c_str());
    size_t cache_size = 0;
    bool dropped = false;
    do {
      dropped = HPUDeviceContext::recipe_cache().drop_lru(cache_size);
    } while (cache_size > 0 && dropped);
    HABANA_ASSERT(HPUDeviceContext::recipe_cache().empty());
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
    HPUDeviceContext::recipe_cache().FlushDiskCache();

    int recipe_files_count = getFilesCount(getCachePath().c_str(), ".recipe");
    if (i == 0) {
      originalRecipe = result;
      size_t one = 1;
      HPUDeviceContext::recipe_cache().drop_lru(one);
      HABANA_ASSERT(HPUDeviceContext::recipe_cache().empty());
      HABANA_ASSERT(recipe_files_count == 1);
    } else if (i == 1) {
      deserializedRecipe = result;
      HABANA_ASSERT(!HPUDeviceContext::recipe_cache().empty());
      HABANA_ASSERT(recipe_files_count == 1);
    }
  }
  auto res2 = deserializedRecipe.to(torch::kCPU);
  auto res1 = originalRecipe.to(torch::kCPU);
  EXPECT_EQ(allclose(res1, res2), true);
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