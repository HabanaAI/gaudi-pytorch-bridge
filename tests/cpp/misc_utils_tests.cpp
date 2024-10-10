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
#include <unordered_set>

#include <gtest/gtest.h>

#include "habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/misc_utils.h"

TEST(MiscUtils, ModExp) {
  std::unordered_set<int64_t> val_set;
  for (int64_t i = -100; i < 100; i++) {
    auto val = habana::mod_exp(i);
    PT_TEST_DEBUG("mod_exp(", i, ") = ", val);
    EXPECT_TRUE(val_set.count(val) == 0);
    val_set.insert(val);
  }
}
