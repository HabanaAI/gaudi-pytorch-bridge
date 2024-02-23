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

#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <c10/util/Exception.h>

#include "backend/synapse_helpers/env_flags.h"
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
