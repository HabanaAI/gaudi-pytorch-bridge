/**
 * Copyright (c) 2025 Intel Corporation
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

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>

#include <gtest/gtest.h>

#include "hpu_ops/autocast_helpers.h"

TEST(TestCustomAutocast, EmptyPath) {
  const std::unordered_set<std::string> default_list = {
      "test_op_1", "test_op_2"};

  const auto list = at::autocast::load_ops_list("", default_list);
  ASSERT_EQ(default_list, list);
}

TEST(TestCustomAutocast, NotExistingFile) {
  const std::unordered_set<std::string> default_list = {
      "test_op_1", "test_op_2"};

  testing::internal::CaptureStderr();
  const auto list =
      at::autocast::load_ops_list("/non-existing-file", default_list);
  const std::string expected_output =
      "[WARNING][PT_BRIDGE][AUTOCAST] Verification of ops list to autocast failed: File(\"/non-existing-file\") does not exist. Loading default ops list.\n";
  const std::string output = testing::internal::GetCapturedStderr();

  ASSERT_EQ(expected_output, output);
  ASSERT_EQ(default_list, list);
}

TEST(TestCustomAutocast, PathToDirectory) {
  const std::unordered_set<std::string> default_list = {
      "test_op_1", "test_op_2"};

  testing::internal::CaptureStderr();
  const auto list = at::autocast::load_ops_list("/home", default_list);
  const std::string expected_output =
      "[WARNING][PT_BRIDGE][AUTOCAST] Verification of ops list to autocast failed: Path(\"/home\") is a directory Loading default ops list.\n";
  const std::string output = testing::internal::GetCapturedStderr();

  ASSERT_EQ(expected_output, output);
  ASSERT_EQ(default_list, list);
}

TEST(TestCustomAutocast, UnsafeLocation) {
  const std::unordered_set<std::string> default_list = {
      "test_op_1", "test_op_2"};

  testing::internal::CaptureStderr();
  const auto list = at::autocast::load_ops_list("/bin/bash", default_list);
  const std::string expected_output =
      "[WARNING][PT_BRIDGE][AUTOCAST] Verification of ops list to autocast failed: Used unsafe file location(\"/bin/bash\"). Loading default ops list.\n";
  const std::string output = testing::internal::GetCapturedStderr();

  ASSERT_EQ(expected_output, output);
  ASSERT_EQ(default_list, list);
}

TEST(TestCustomAutocast, IsOperatorCorrect) {
  // Generate a temporary file for tests.
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> sufix(1000, 9999);

  std::stringstream temporary_file_name;
  temporary_file_name << std::tmpnam(nullptr) << sufix(generator) << ".txt";
  std::filesystem::path temp_file_path(temporary_file_name.str());

  // Fill file with test ops
  std::ofstream temp_file(temp_file_path);
  temp_file << "div" << std::endl
            << "add" << std::endl
            << "invalid_op" << std::endl;
  temp_file.close();

  const std::unordered_set<std::string> default_list = {};
  const std::unordered_set<std::string> expected_list = {"div", "add"};

  const auto list = at::autocast::load_ops_list(temp_file_path, default_list);

  ASSERT_EQ(expected_list, list);

  std::filesystem::remove(temp_file_path);
}
