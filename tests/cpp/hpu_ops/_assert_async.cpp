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

#include "../utils/device_type_util.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "util.h"

using namespace habana_lazy;
using namespace at;

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, AsyncAssert) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  auto y = torch::zeros(1).to(torch::kHPU);
  auto z = torch::zeros(1).to(torch::kHPU);
  auto x = torch::eq(y, z);
  torch::_assert_async(x);
  HbLazyTensor::StepMarker({});
}