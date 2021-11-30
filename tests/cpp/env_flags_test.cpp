/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

TEST(EnvFlags, GetEnv) {
  PT_TEST_DEBUG(
      "PT_HPU_LAZY_MODE ",
      (IS_ENV_FLAG_DEFINED_NEW(PT_HPU_LAZY_MODE) ? "defined" : "not defined"));

  // auto env_val_old = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  // PT_TEST_DEBUG("OLD PT_HPU_LAZY_MODE=", env_val);

  auto env_val = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  PT_TEST_DEBUG("PT_HPU_LAZY_MODE=", env_val);
  EXPECT_EQ(env_val, 2);

  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 1, 1);

  env_val = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  PT_TEST_DEBUG("PT_HPU_LAZY_MODE=", env_val);
  EXPECT_EQ(env_val, 1);

  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 0, 1);

  env_val = GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
  PT_TEST_DEBUG("PT_HPU_LAZY_MODE=", env_val);
  EXPECT_EQ(env_val, 0);

  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
}

TEST(EnvFlags, FatalMessage) {
  auto env_val_typ = GET_ENV_FLAG(PT_HPU_LOG_TYPE_MASK);
  auto env_val_mod = GET_ENV_FLAG(PT_HPU_LOG_MOD_MASK);

  std::stringstream ss;
  ss << "0x" << std::uppercase << std::hex << env_val_typ;
  std::string env_val_typ_str{ss.str()};
  ss.str(std::string{});
  ss << "0x" << std::uppercase << std::hex << env_val_mod;
  std::string env_val_mod_str{ss.str()};

  auto restore_env{[&]() {
    PT_TEST_DEBUG(
        "Restoring logging env setup : "
        "PT_HPU_LOG_TYPE_MASK=",
        env_val_typ_str,
        "  PT_HPU_LOG_MOD_MASK=",
        env_val_mod_str);

    setenv("PT_HPU_LOG_TYPE_MASK", env_val_typ_str.c_str(), 1);
    setenv("PT_HPU_LOG_MOD_MASK", env_val_mod_str.c_str(), 1);

    // Refresh the logger
    PtLogger::getLogger()->refresh();
  }};

  PT_TEST_DEBUG(
      "Initial logging env setup : "
      "PT_HPU_LOG_TYPE_MASK=",
      env_val_typ_str,
      "  PT_HPU_LOG_MOD_MASK=",
      env_val_mod_str);

  std::set<std::string> test_set{
      "", "0", "0x0", "1234", "0x1234", "AbCd", "0xAbCd"};

  for (auto env_str : test_set) {
    if (env_str.empty()) {
      PT_TEST_DEBUG(
          "New logging env setup : "
          "unset PT_HPU_LOG_TYPE_MASK",
          ", unset PT_HPU_LOG_MOD_MASK");

      unsetenv("PT_HPU_LOG_TYPE_MASK");
      unsetenv("PT_HPU_LOG_MOD_MASK");
    } else {
      PT_TEST_DEBUG(
          "New logging env setup : "
          "PT_HPU_LOG_TYPE_MASK=",
          env_str,
          "  PT_HPU_LOG_MOD_MASK=",
          env_str);

      setenv("PT_HPU_LOG_TYPE_MASK", env_str.c_str(), 1);
      setenv("PT_HPU_LOG_MOD_MASK", env_str.c_str(), 1);
    }

    // Refresh the logger
    PtLogger::getLogger()->refresh();

    GET_ENV_FLAG(PT_HPU_LOG_TYPE_MASK);
    GET_ENV_FLAG(PT_HPU_LOG_MOD_MASK);

    try {
      PT_SYNHELPER_FATAL("<example error message>");

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

TEST(EnvFlags, Logging) {
  auto env_val_typ = GET_ENV_FLAG(PT_HPU_LOG_TYPE_MASK);
  auto env_val_mod = GET_ENV_FLAG(PT_HPU_LOG_MOD_MASK);

  std::stringstream ss;
  ss << "0x" << std::uppercase << std::hex << env_val_typ;
  std::string env_val_typ_str{ss.str()};
  ss.str(std::string{});
  ss << "0x" << std::uppercase << std::hex << env_val_mod;
  std::string env_val_mod_str{ss.str()};

  auto restore_env{[&]() {
    PT_TEST_DEBUG(
        "Restoring logging env setup : "
        "PT_HPU_LOG_TYPE_MASK=",
        env_val_typ_str,
        "  PT_HPU_LOG_MOD_MASK=",
        env_val_mod_str);

    setenv("PT_HPU_LOG_TYPE_MASK", env_val_typ_str.c_str(), 1);
    setenv("PT_HPU_LOG_MOD_MASK", env_val_mod_str.c_str(), 1);

    // Refresh the logger
    PtLogger::getLogger()->refresh();
  }};

  PT_TEST_DEBUG(
      "Initial logging env setup : "
      "PT_HPU_LOG_TYPE_MASK=",
      env_val_typ_str,
      "  PT_HPU_LOG_MOD_MASK=",
      env_val_mod_str);

  std::set<std::pair<std::string, std::string>> test_set{
      {"1234", "fffffffffffffffff"}, {"ff", "a1234sd"}};

  for (auto p : test_set) {
    auto typ_str{p.first};
    auto mod_str{p.second};

    PT_TEST_DEBUG(
        "New logging env setup : "
        "PT_HPU_LOG_TYPE_MASK=",
        typ_str,
        "  PT_HPU_LOG_MOD_MASK=",
        mod_str);

    setenv("PT_HPU_LOG_TYPE_MASK", typ_str.c_str(), 1);
    setenv("PT_HPU_LOG_MOD_MASK", mod_str.c_str(), 1);

    try {
      // Refresh the logger
      PtLogger::getLogger()->refresh();

      GET_ENV_FLAG(PT_HPU_LOG_TYPE_MASK);
      GET_ENV_FLAG(PT_HPU_LOG_MOD_MASK);

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
