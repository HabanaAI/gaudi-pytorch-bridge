/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************
 */

#include "../utils/device_type_util.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "util.h"

using namespace habana_lazy;
using namespace at;

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, AsyncAssert) {
  if (isGaudi3() || isGaudi2()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  auto y = torch::zeros(1).to(torch::kHPU);
  auto z = torch::zeros(1).to(torch::kHPU);
  auto x = torch::eq(y, z);
  torch::_assert_async(x);
  HbLazyTensor::StepMarker({});
}