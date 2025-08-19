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

#include <set>
#include <string>

#include <gtest/gtest.h>
#include <stdexcept>

#include <filesystem>
#include "habana_helpers/logging.h"

TEST(EnvFlags, GetEnv) {
  PT_TEST_DEBUG(
      "PT_HPU_LAZY_MODE ",
      (IS_ENV_FLAG_DEFINED_NEW(PT_HPU_LAZY_MODE) ? "defined" : "not defined"));

  auto is_env_val_org_defined = IS_ENV_FLAG_DEFINED_NEW(PT_HPU_LAZY_MODE);
  auto env_val_org = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);

  // Unset env variable
  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  auto is_env_val_defined = IS_ENV_FLAG_DEFINED_NEW(PT_HPU_LAZY_MODE);
  EXPECT_EQ(is_env_val_defined, false);

  // Env var not defined get default value
  auto env_val = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  PT_TEST_DEBUG("PT_HPU_LAZY_MODE=", env_val);
  EXPECT_EQ(env_val, 1);

  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 1, 1);

  env_val = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  PT_TEST_DEBUG("PT_HPU_LAZY_MODE=", env_val);
  EXPECT_EQ(env_val, 1);

  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 0, 1);

  env_val = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  PT_TEST_DEBUG("PT_HPU_LAZY_MODE=", env_val);
  EXPECT_EQ(env_val, 0);

  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);

  if (is_env_val_org_defined) {
    SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, env_val_org, 1);
    PT_TEST_DEBUG("Restore original PT_HPU_LAZY_MODE=", env_val_org);
  }

  // Test string env variables
  PT_TEST_DEBUG(
      "PT_HPU_GRAPH_DUMP_PREFIX ",
      (IS_ENV_FLAG_DEFINED_NEW(PT_HPU_GRAPH_DUMP_PREFIX) ? "defined"
                                                         : "not defined"));

  auto is_env_str_val_org_defined =
      IS_ENV_FLAG_DEFINED_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  auto env_str_val_org = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);

  // Unset env str variable
  UNSET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  auto is_env_str_val_defined =
      IS_ENV_FLAG_DEFINED_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  EXPECT_EQ(is_env_str_val_defined, false);

  // Env str var not defined get default value
  std::string env_str_val = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  PT_TEST_DEBUG("PT_HPU_GRAPH_DUMP_PREFIX=", env_str_val);
  EXPECT_EQ(env_str_val, ".graph_dumps");

  SET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX, "./tmp_path", 1);

  env_str_val = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  PT_TEST_DEBUG("PT_HPU_GRAPH_DUMP_PREFIX=", env_str_val);
  EXPECT_EQ(env_str_val, "./tmp_path");

  UNSET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);

  if (is_env_str_val_org_defined) {
    SET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX, env_str_val_org, 1);
    PT_TEST_DEBUG(
        "Restore original PT_HPU_GRAPH_DUMP_PREFIX=", env_str_val_org);
  }

  // Test cache skip env variables
  PT_TEST_DEBUG(
      "PT_ENABLE_HCL_STREAM ",
      (IS_ENV_FLAG_DEFINED_NEW(PT_ENABLE_HCL_STREAM) ? "defined"
                                                     : "not defined"));

  // Get default value to restore it back if env var is defined.
  is_env_val_org_defined = IS_ENV_FLAG_DEFINED_NEW(PT_ENABLE_HCL_STREAM);
  env_val_org = GET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM);

  // Unset the cached env flag
  UNSET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM);
  auto is_env_bool_val_defined = IS_ENV_FLAG_DEFINED_NEW(PT_ENABLE_HCL_STREAM);
  EXPECT_EQ(is_env_bool_val_defined, false);

  // Set non "string" value to bool env var
  setenv("PT_ENABLE_HCL_STREAM", "1", 1);

  // Get the cached env value
  bool env_bool_val = GET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM);
  PT_TEST_DEBUG("PT_ENABLE_HCL_STREAM=", env_bool_val);
  EXPECT_EQ(env_bool_val, true);

  // Set non "string" value to bool env var
  setenv("PT_ENABLE_HCL_STREAM", "0", 1);

  // Get cached env value with skip_cache disable
  env_bool_val = GET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM, false);
  PT_TEST_DEBUG("PT_ENABLE_HCL_STREAM=", env_bool_val);
  EXPECT_EQ(env_bool_val, true);

  // Get system env value with skip_cache enable
  env_bool_val = GET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM, true);
  PT_TEST_DEBUG("PT_ENABLE_HCL_STREAM=", env_bool_val);
  EXPECT_EQ(env_bool_val, false);

  // Unset the cached env flag
  UNSET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM);

  if (is_env_bool_val_defined) {
    SET_ENV_FLAG_NEW(PT_ENABLE_HCL_STREAM, env_val_org, 1);
    PT_TEST_DEBUG("Restore original PT_ENABLE_HCL_STREAM=", env_val_org);
  }
  // unset env flag
  unsetenv("PT_ENABLE_HCL_STREAM");
}

TEST(EnvFlags, FatalMessage) {
  auto env_val = getenv("LOG_LEVEL_PT_DEVICE");
  std::string env_val_str;
  if (env_val)
    env_val_str = std::string(env_val);

  auto restore_env{[&]() {
    PT_TEST_DEBUG(
        "Restoring logging env setup : "
        "LOG_LEVEL_PT_DEVICE=",
        env_val_str);

    unsetenv("LOG_LEVEL_PT_DEVICE");
    setenv("LOG_LEVEL_PT_DEVICE", env_val_str.c_str(), 1);
  }};

  PT_TEST_DEBUG(
      "Initial logging env setup : "
      "LOG_LEVEL_PT_DEVICE=",
      env_val_str);

  std::set<std::string> test_set{"", "0", "1", "10"};

  for (auto env_str : test_set) {
    if (env_str.empty()) {
      PT_TEST_DEBUG("New logging env setup : unset LOG_LEVEL_PT_DEVICE");
    } else {
      PT_TEST_DEBUG(
          "New logging env setup : "
          "LOG_LEVEL_PT_DEVICE=",
          env_str);

      unsetenv("LOG_LEVEL_PT_DEVICE");
      setenv("LOG_LEVEL_PT_DEVICE", env_str.c_str(), 1);
    }

    try {
      PT_DEVICE_FATAL("<example error message>");

      // If it fails to raise an exception, we should restore the env
      restore_env();
    } catch (const std::exception& e) {
      // restore the env before we do anything else
      restore_env();
      try {
        auto& act_excp =
            dynamic_cast<c10::Error&>(const_cast<std::exception&>(e));
        PT_TEST_DEBUG(
            "Caught ", act_excp.what(), "Exception raised as per expectation");
      } catch (std::bad_cast& bc) {
        PT_TEST_DEBUG("Caught bad_cast : ", bc.what());
        FAIL() << "Unknown exception received";
      }
    } catch (...) {
      PT_TEST_DEBUG("Caught UNFATAL exception");
      FAIL() << "Unknown exception encountered";
    }
  }
}

TEST(UserEnvFlagsValidation, CheckEnum) {
  auto env_val_org = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  // Define a mock environment flag for testing enum validation and type was
  // unsigned
  const char* env_name = "PT_HPU_LAZY_MODE";
  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  setenv(env_name, "1", 1); // Valid value
  auto env_val = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  PT_TEST_DEBUG("PT_HPU_LAZY_MODE=", env_val);
  EXPECT_EQ(env_val, 1);
  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  setenv(env_name, "3", 1); // Invalid value
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_HPU_LAZY_MODE,Value '3' is not allowed. Allowed values: 0 | 1") !=
              std::string::npos);
          throw;
        }
      },
      c10::Error);
  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);

  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, env_val_org, 1);
  PT_TEST_DEBUG("Restore original PT_HPU_LAZY_MODE=", env_val_org);

  PT_TEST_DEBUG("Unset env PT_HPU_LAZY_MODE");
  // unset env flag
  unsetenv("PT_HPU_LAZY_MODE");

  // Define a mock environment flag for testing enum validation and type was
  // string
  std::string env_val_org_dump_mode = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE);
  UNSET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE);

  env_name = "PT_HPU_GRAPH_DUMP_MODE";
  setenv(env_name, "compile", 1); // Valid value
  std::string env_val_str = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE);
  PT_TEST_DEBUG("PT_HPU_GRAPH_DUMP_MODE=", env_val_str);
  EXPECT_EQ(env_val_str, "compile");
  UNSET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE);

  // Invalid enum value
  setenv(env_name, "compile_fx", 1);
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_HPU_GRAPH_DUMP_MODE,Value 'compile_fx' is not allowed. Allowed values: compile | eager | all") !=
              std::string::npos);
          throw;
        }
      },
      c10::Error);
  UNSET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE);
  SET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_MODE, env_val_org_dump_mode.c_str(), 1);
  PT_TEST_DEBUG(
      "Restore original PT_HPU_GRAPH_DUMP_MODE=", env_val_org_dump_mode);

  PT_TEST_DEBUG("Unset env PT_HPU_GRAPH_DUMP_MODE");
  // unset env flag
  unsetenv("PT_HPU_GRAPH_DUMP_MODE");
}

TEST(UserEnvFlagsValidation, CheckFilepath) {
  std::string env_val_org = GET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
  // Define a mock environment flag for testing filepath validation and if the
  // path not exists needs to be created and the path was valid
  const char* env_name = "PT_RECIPE_TRACE_PATH";
  UNSET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
  setenv(env_name, "/tmp/recipe_trace_path", 1); // Valid value
  std::string env_val = GET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
  PT_TEST_DEBUG("PT_RECIPE_TRACE_PATH=", env_val);
  EXPECT_EQ(env_val, "/tmp/recipe_trace_path");
  UNSET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);

  // Invalid path with unsafe characters
  setenv(env_name, "/tmp/invalid|path", 1);
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_RECIPE_TRACE_PATH,Value contains insecure characters: /tmp/invalid|path") !=
              std::string::npos);
          throw;
        }
      },
      c10::Error);

  UNSET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
  // Invalid path with insecure characters
  setenv(env_name, "/tmp/insecure$path", 1);
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_RECIPE_TRACE_PATH,Value contains insecure characters: /tmp/insecure$path") !=
              std::string::npos);
          throw;
        }
      },
      c10::Error);

  UNSET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH);
  SET_ENV_FLAG_NEW(PT_RECIPE_TRACE_PATH, env_val_org.c_str(), 1);
  PT_TEST_DEBUG("Restore original PT_RECIPE_TRACE_PATH=", env_val_org);

  PT_TEST_DEBUG("Unset env PT_RECIPE_TRACE_PATH");
  // unset env flag
  unsetenv("PT_RECIPE_TRACE_PATH");

  env_val_org = GET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);
  UNSET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);

  // Non-existent path without creation needed
  env_name = "PT_COMPILATION_STATS_PATH";
  setenv(env_name, "/nonexistent/path", 1);
  EXPECT_NO_THROW({
    try {
      GET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);
    } catch (const std::exception& e) {
      FAIL() << "Unexpected exception: " << e.what();
    }
  });
  UNSET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH);
  SET_ENV_FLAG_NEW(PT_COMPILATION_STATS_PATH, env_val_org.c_str(), 1);
  PT_TEST_DEBUG("Restore original PT_COMPILATION_STATS_PATH=", env_val_org);

  PT_TEST_DEBUG("Unset env PT_COMPILATION_STATS_PATH");
  // unset env flag
  unsetenv("PT_COMPILATION_STATS_PATH");
}

TEST(UserEnvFlagsValidation, CheckRecipeCacheConfig) {
  std::string env_val_org = GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  const char* env_name = "PT_HPU_RECIPE_CACHE_CONFIG";
  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  // Create tmp directory for testing
  std::filesystem::create_directories("/tmp/path");
  setenv(env_name, "/tmp/path,true,1024,false", 1); // Valid value
  std::string env_val = GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  PT_TEST_DEBUG("PT_HPU_RECIPE_CACHE_CONFIG=", env_val);
  EXPECT_EQ(env_val, "/tmp/path,true,1024,false");

  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  // Missing one field
  setenv(env_name, "/tmp/path,true,1024", 1);
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_HPU_RECIPE_CACHE_CONFIG,/tmp/path,true,1024 does not match expected format. Expected: <path> or <path>,<true|false>,<int_size_in_MB>,<true|false>.") !=
              std::string::npos);
          throw;
        }
      },
      c10::Error);

  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  // Non-existent directory
  setenv(env_name, "/nonexistent/path,true,1024,false", 1);
  EXPECT_NO_THROW({
    try {
      GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
    } catch (const std::exception& e) {
      FAIL() << "Unexpected exception: " << e.what();
    }
  });

  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  // Path is not a directory
  // Create a temporary file to test "Path is not a directory" case
  std::string test_file = "/tmp/path/file";
  std::ofstream ofs(test_file);
  ASSERT_TRUE(ofs.is_open());
  ofs << "dummy content\n";
  ofs.close();
  setenv(
      env_name, "/tmp/path/file,true,1024,false", 1); // Path is not a directory
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Path is not a directory: /tmp/path/file") !=
              std::string::npos);
          throw;
        }
      },
      c10::Error);

  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  // Invalid clear_on_init value
  setenv(env_name, "/tmp/path,invalid,1024,false", 1);
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_HPU_RECIPE_CACHE_CONFIG,/tmp/path,invalid,1024,false does not match expected format. Expected: <path> or <path>,<true|false>,<int_size_in_MB>,<true|false>.") !=
              std::string::npos);

          throw;
        }
      },
      c10::Error);

  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  // Invalid cache size
  setenv(env_name, "/tmp/path,true,invalid,false", 1);
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_HPU_RECIPE_CACHE_CONFIG,/tmp/path,true,invalid,false does not match expected format. Expected: <path> or <path>,<true|false>,<int_size_in_MB>,<true|false>.") !=
              std::string::npos);

          throw;
        }
      },
      c10::Error);

  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  // Invalid NFS flag value
  setenv(env_name, "/tmp/path,true,1024,invalid", 1);
  EXPECT_THROW(
      {
        try {
          GET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "Flags name: PT_HPU_RECIPE_CACHE_CONFIG,/tmp/path,true,1024,invalid does not match expected format. Expected: <path> or <path>,<true|false>,<int_size_in_MB>,<true|false>.") !=
              std::string::npos);

          throw;
        }
      },
      c10::Error);

  UNSET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG);
  SET_ENV_FLAG_NEW(PT_HPU_RECIPE_CACHE_CONFIG, env_val_org.c_str(), 1);
  PT_TEST_DEBUG("Restore original PT_HPU_RECIPE_CACHE_CONFIG=", env_val_org);

  PT_TEST_DEBUG("Unset env PT_HPU_RECIPE_CACHE_CONFIG");
  // unset env flag
  unsetenv("PT_HPU_RECIPE_CACHE_CONFIG");
}

#include "backend/synapse_helpers/env_flags_impl.cpp"
TEST(FunctionEnvFlagsValidation, CheckEnum) {
  std::string errorMsg;

  // Valid enum value
  EXPECT_TRUE(
      env_flags::new_style::check_enum(
          "value1", "value1|value2|value3", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Invalid enum value
  EXPECT_FALSE(
      env_flags::new_style::check_enum(
          "value4", "value1|value2|value3", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "Value 'value4' is not allowed. Allowed values: value1|value2|value3.");
  errorMsg.clear();

  // Valid enum value with spaces around constraints
  EXPECT_TRUE(
      env_flags::new_style::check_enum(
          "value2", " value1 | value2 | value3 ", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Invalid format for constraints
  EXPECT_FALSE(
      env_flags::new_style::check_enum("value1", "value1|value2|", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "value1|value2| does not match expected format. Expected: value1 | value2 | value3");
  errorMsg.clear();

  // Empty constraints
  EXPECT_FALSE(env_flags::new_style::check_enum("value1", "", errorMsg));
  EXPECT_EQ(
      errorMsg,
      " does not match expected format. Expected: value1 | value2 | value3");
  errorMsg.clear();

  // Empty value
  EXPECT_FALSE(
      env_flags::new_style::check_enum("", "value1|value2|value3", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "Value '' is not allowed. Allowed values: value1|value2|value3.");
}

TEST(FunctionEnvFlagsValidation, CheckRange) {
  std::string errorMsg;

  // Valid range and value
  EXPECT_TRUE(env_flags::new_style::check_range("5", "(1, 10)", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Value at the lower bound of the range
  EXPECT_TRUE(env_flags::new_style::check_range("1", "(1, 10)", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Value at the upper bound of the range
  EXPECT_TRUE(env_flags::new_style::check_range("10", "(1, 10)", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Value below the range
  EXPECT_FALSE(env_flags::new_style::check_range("0", "(1, 10)", errorMsg));
  EXPECT_EQ(errorMsg, "Value '0' is out of range. Allowed range: (1, 10)");
  errorMsg.clear();

  // Value above the range
  EXPECT_FALSE(env_flags::new_style::check_range("11", "(1, 10)", errorMsg));
  EXPECT_EQ(errorMsg, "Value '11' is out of range. Allowed range: (1, 10)");
  errorMsg.clear();

  // Invalid range format
  EXPECT_FALSE(env_flags::new_style::check_range("5", "(1,10", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "(1,10 does not match expected format. Expected: (start, end), e.g. (1, 10)");
  errorMsg.clear();

  // Invalid value (non-integer)
  EXPECT_FALSE(env_flags::new_style::check_range("abc", "(1, 10)", errorMsg));
  EXPECT_EQ(errorMsg, "Value 'abc' is not a valid integer.");
  errorMsg.clear();

  // Negative range and value
  EXPECT_TRUE(env_flags::new_style::check_range("-5", "(-10, -1)", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Value outside negative range
  EXPECT_FALSE(env_flags::new_style::check_range("-15", "(-10, -1)", errorMsg));
  EXPECT_EQ(errorMsg, "Value '-15' is out of range. Allowed range: (-10, -1)");
}

TEST(FunctionEnvFlagsValidation, CheckList) {
  std::string errorMsg;

  // Valid ordered list
  EXPECT_TRUE(
      env_flags::new_style::check_list(
          "4,5,3,1", "ordered=true,<3,1>", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Valid lower suffix of ordered
  EXPECT_TRUE(
      env_flags::new_style::check_list(
          "4,5,3,1", "ordered=True,<3,1>", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Invalid ordered list (does not match suffix)
  EXPECT_FALSE(
      env_flags::new_style::check_list(
          "4,5,1,3", "ordered=true,<3,1>", errorMsg));
  EXPECT_EQ(
      errorMsg, "Input list does not match required ordered suffix: <3,1>.");
  errorMsg.clear();

  // Valid unordered list
  EXPECT_TRUE(
      env_flags::new_style::check_list(
          "4,5", "ordered=false,<4,5,3,1>", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Invalid unordered list (contains disallowed item)
  EXPECT_FALSE(
      env_flags::new_style::check_list(
          "4,6", "ordered=false,<4,5,3,1>", errorMsg));
  EXPECT_EQ(errorMsg, "Input item '6' is not in allowed set: <4,5,3,1>.");
  errorMsg.clear();

  // Input list shorter than allowed suffix
  EXPECT_FALSE(
      env_flags::new_style::check_list("3", "ordered=true,<3,1>", errorMsg));
  EXPECT_EQ(errorMsg, "Input list is shorter than allowed suffix.");
  errorMsg.clear();

  // Invalid format for constraints
  EXPECT_FALSE(
      env_flags::new_style::check_list(
          "4,5,3,1", "ordered=true,<3,1", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "ordered=true,<3,1 does not match expected format. Expected: ordered=true|false,<A,B,C>.");
  errorMsg.clear();

  // Missing 'ordered' part in constraints
  EXPECT_FALSE(env_flags::new_style::check_list("4,5,3,1", "<3,1>", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "<3,1> does not match expected format. Expected: ordered=true|false,<A,B,C>.");
  errorMsg.clear();

  // Empty input list
  EXPECT_FALSE(
      env_flags::new_style::check_list(
          "", "ordered=false,<4,5,3,1>", errorMsg));
  EXPECT_EQ(errorMsg, "Input list is empty.");
  errorMsg.clear();

  // input and constraints with spaces
  EXPECT_TRUE(
      env_flags::new_style::check_list(
          " 4 , 5 ", " ordered = false , < 4 , 5 , 3 , 1 >", errorMsg));
  EXPECT_TRUE(errorMsg.empty());
}

TEST(FunctionEnvFlagsValidation, CheckFilepath) {
  std::string errorMsg;

  // Valid path that exists
  std::filesystem::create_directories("/tmp/filepath_test");
  EXPECT_TRUE(
      env_flags::new_style::check_file_path(
          "/tmp/filepath_test", "is_dir & is_create=false", errorMsg));
  EXPECT_TRUE(errorMsg.empty());
  std::filesystem::remove_all("/tmp/filepath_test");

  // Valid path with creation allowed
  EXPECT_TRUE(
      env_flags::new_style::check_file_path(
          "/tmp/new_directory", "is_dir & is_create=true", errorMsg));
  EXPECT_TRUE(errorMsg.empty());
  EXPECT_TRUE(std::filesystem::exists("/tmp/new_directory"));
  std::filesystem::remove_all("/tmp/new_directory");

  // Invalid path with unsafe characters
  EXPECT_FALSE(
      env_flags::new_style::check_file_path(
          "/tmp/invalid|path", "is_dir & is_create=false", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "Directory path contains unsafe characters: /tmp/invalid|path. Please avoid using characters like [\\s\"'\\\\<>|&;$%*?\\[\\]\\{\\}^~`] in the path.");
  errorMsg.clear();

  // Non-existent path without creation needed
  EXPECT_TRUE(
      env_flags::new_style::check_file_path(
          "/nonexistent/path", "is_dir & is_create=false", errorMsg));
  EXPECT_TRUE(errorMsg.empty());
  errorMsg.clear();

  // Path exists but is not a directory
  std::string test_file = "/tmp/test_file";
  std::ofstream ofs(test_file);
  ASSERT_TRUE(ofs.is_open());
  ofs << "dummy content\n";
  ofs.close();
  EXPECT_FALSE(
      env_flags::new_style::check_file_path(
          test_file, "is_dir & is_create=false", errorMsg));
  EXPECT_EQ(errorMsg, "Path is not a directory: /tmp/test_file");
  std::remove(test_file.c_str());
  errorMsg.clear();

  // Path is a symbolic link
  std::filesystem::create_directories("/tmp/real_directory");
  std::filesystem::create_symlink(
      "/tmp/real_directory", "/tmp/symlink_directory");
  EXPECT_FALSE(
      env_flags::new_style::check_file_path(
          "/tmp/symlink_directory", "is_dir & is_create=false", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "Path is a symbolic link: /tmp/symlink_directory. Symbolic links are not allowed.");
  std::filesystem::remove("/tmp/symlink_directory");
  std::filesystem::remove_all("/tmp/real_directory");
  errorMsg.clear();

  // Directory is not writable
  std::filesystem::create_directories("/tmp/not_writable_directory");
  std::filesystem::permissions(
      "/tmp/not_writable_directory", std::filesystem::perms::owner_read);
  EXPECT_FALSE(
      env_flags::new_style::check_file_path(
          "/tmp/not_writable_directory", "is_dir & is_create=false", errorMsg));
  EXPECT_EQ(
      errorMsg, "Directory path is not writable: /tmp/not_writable_directory");
  std::filesystem::permissions(
      "/tmp/not_writable_directory", std::filesystem::perms::owner_all);
  std::filesystem::remove_all("/tmp/not_writable_directory");
  errorMsg.clear();
}

TEST(FunctionEnvFlagsValidation, CheckRecipeCacheConfig) {
  std::string errorMsg;
  // Create tmp directory for testing
  std::filesystem::create_directories("/tmp/recipe-cache");
  // Valid configuration
  EXPECT_TRUE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/recipe-cache,true,1024,false", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // Invalid format: missing fields
  EXPECT_FALSE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/recipe-cache,true,1024", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "/tmp/recipe-cache,true,1024 does not match expected format. Expected: <path> or <path>,<true|false>,<int_size_in_MB>,<true|false>.");
  errorMsg.clear();

  // Invalid directory path: contains unsafe characters
  EXPECT_FALSE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/invalid|path,true,1024,false", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "/tmp/invalid|path,true,1024,false does not match expected format. Expected: <path> or <path>,<true|false>,<int_size_in_MB>,<true|false>.");
  errorMsg.clear();

  // Invalid directory path: contains unsafe characters
  EXPECT_FALSE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/unsafe$path,true,1024,false", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "Directory path contains unsafe characters: /tmp/unsafe$path. Please avoid using characters like [\\s\"'\\\\<>|&;$%*?\\[\\]\\{\\}^~`] in the path.");
  errorMsg.clear();

  // valid directory path: does not exist, it will just warning and the process
  // continues
  EXPECT_NO_THROW({
    try {
      EXPECT_TRUE(
          env_flags::new_style::check_recipe_cache_config(
              "/nonexistent/path,true,1024,false", errorMsg));
    } catch (const std::exception& e) {
      FAIL() << "Unexpected exception: " << e.what();
    }
  });
  errorMsg.clear();

  // Invalid directory path: not a directory
  std::string test_file = "/tmp/test_file";
  std::ofstream ofs(test_file);
  ASSERT_TRUE(ofs.is_open());
  ofs << "dummy content\n";
  ofs.close();
  EXPECT_FALSE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/test_file,true,1024,false", errorMsg));
  EXPECT_EQ(errorMsg, "Path is not a directory: /tmp/test_file");
  std::remove(test_file.c_str());
  errorMsg.clear();

  // Path is a symbolic link
  std::filesystem::create_directories("/tmp/real_directory");
  std::filesystem::create_symlink(
      "/tmp/real_directory", "/tmp/symlink_directory");
  EXPECT_FALSE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/symlink_directory,true,1024,false", errorMsg));
  EXPECT_EQ(
      errorMsg,
      "Path is a symbolic link: /tmp/symlink_directory. Symbolic links are not allowed.");
  std::filesystem::remove("/tmp/symlink_directory");
  std::filesystem::remove_all("/tmp/real_directory");
  errorMsg.clear();

  // Directory is not writable
  std::filesystem::create_directories("/tmp/not_writable_directory");
  std::filesystem::permissions(
      "/tmp/not_writable_directory", std::filesystem::perms::owner_read);
  EXPECT_FALSE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/not_writable_directory,true,1024,false", errorMsg));
  EXPECT_EQ(
      errorMsg, "Directory path is not writable: /tmp/not_writable_directory");
  std::filesystem::permissions(
      "/tmp/not_writable_directory", std::filesystem::perms::owner_all);
  std::filesystem::remove_all("/tmp/not_writable_directory");
  errorMsg.clear();

  // Valid configuration with empty directory path
  EXPECT_TRUE(
      env_flags::new_style::check_recipe_cache_config(
          ",true,1024,false", errorMsg));
  EXPECT_TRUE(errorMsg.empty());

  // valid input config with spaces around values
  EXPECT_TRUE(
      env_flags::new_style::check_recipe_cache_config(
          "/tmp/recipe-cache, true ,1024,false", errorMsg));
  EXPECT_TRUE(errorMsg.empty());
  std::filesystem::remove_all("/tmp/recipe-cache");
}

TEST(FunctionEnvFlagsValidation, CheckFlagStatus) {
  // Test case for FLAG_STATUS_DEPRECATED
  EXPECT_NO_THROW({
    try {
      check_flag_status("TEST_FLAG_DEPRECATED", FLAG_STATUS_DEPRECATED);
    } catch (const std::exception& e) {
      FAIL() << "Unexpected exception: " << e.what();
    }
  });

  // Test case for FLAG_STATUS_OBSOLETE
  EXPECT_THROW(
      {
        try {
          check_flag_status("TEST_FLAG_OBSOLETE", FLAG_STATUS_OBSOLETE);
        } catch (const c10::Error& e) {
          EXPECT_TRUE(
              std::string(e.what()).find(
                  "[Config Error] Environment flag TEST_FLAG_OBSOLETE is obsolete and cannot be set.") !=
              std::string::npos);
          throw;
        }
      },
      c10::Error);
}

class EnvFlagsTestFixture : public ::testing::TestWithParam<int> {};

TEST_P(EnvFlagsTestFixture, StringTest) {
  std::string val_0 = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  int run_instance = GetParam();
  if (!run_instance) // first run
    ASSERT_EQ(val_0, ".graph_dumps");
  else // second and last run
    ASSERT_EQ(val_0, ".tmp_prefix_1_");

  std::string prefix_1{".tmp_prefix_1_"};
  SET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX, prefix_1.c_str(), 1);
  std::string val_1 = GET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
  ASSERT_EQ(val_1, prefix_1);

  if (run_instance) { // second and last run
    UNSET_ENV_FLAG_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
    auto is_env_defined = IS_ENV_FLAG_DEFINED_NEW(PT_HPU_GRAPH_DUMP_PREFIX);
    ASSERT_EQ(is_env_defined, false);
  }
}

INSTANTIATE_TEST_CASE_P(
    Instantiation,
    EnvFlagsTestFixture,
    ::testing::Range(0, 2));
