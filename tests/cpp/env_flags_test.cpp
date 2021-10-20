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

#include <iostream>

#include <gtest/gtest.h>

#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"

TEST(EnvFlagsTest, GetEnv) {
  PT_TEST_DEBUG(
      "PT_HPU_LAZY_MODE ",
      (IS_ENV_FLAG_DEFINED_NEW(PT_HPU_LAZY_MODE) ? "defined" : "not defined"));

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
