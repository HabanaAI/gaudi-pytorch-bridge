/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include <tests/cpp/habana_lazy_test_infra.h>
#include "common_functions_custom_kernel_tests.h"

class EagerCustomKernelTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode(2);
  }
  void TearDown() override {
    RestoreMode();
  }
};

LAMB_PHASE2_OPT_TEST(EagerCustomKernelTest)
LARS_OPT_TEST(EagerCustomKernelTest, false)
RESOURCE_APPLY_MOMENTUM_OPT_TEST(EagerCustomKernelTest, false)
