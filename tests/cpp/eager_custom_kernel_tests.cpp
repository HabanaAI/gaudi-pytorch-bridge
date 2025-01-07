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
#include <tests/cpp/habana_lazy_test_infra.h>
#include "common_functions_custom_kernel_tests.h"

class DISABLED_EagerCustomKernelTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode(2);
  }
  void TearDown() override {
    RestoreMode();
  }
};

EMA_OPT_TEST(DISABLED_EagerCustomKernelTest, false)
LAMB_PHASE1_OPT_TEST(DISABLED_EagerCustomKernelTest, false)
LAMB_PHASE2_OPT_TEST(DISABLED_EagerCustomKernelTest)
LARS_OPT_TEST(DISABLED_EagerCustomKernelTest, false)
RESOURCE_APPLY_MOMENTUM_OPT_TEST(DISABLED_EagerCustomKernelTest, false)
