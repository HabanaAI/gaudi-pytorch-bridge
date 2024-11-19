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
