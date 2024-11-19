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

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyExecutionTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyExecutionTest, testmultipleLaunch) {
  for (int i = 0; i < 50; i++) {
    auto in =
        torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
    auto wt = torch::randn({4, 5, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
    auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
    auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);

    torch::Tensor result =
        torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
    HbLazyTensor::StepMarker({});
  }
}

TEST_F(LazyExecutionTest, testmultipleLaunchwithAsync) {
  for (int i = 0; i < 50; i++) {
    auto in =
        torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
    auto wt = torch::randn({4, 5, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
    auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
    auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

    auto h_in = in.to(torch::kHPU);
    auto h_wt = wt.to(torch::kHPU);

    torch::Tensor result =
        torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
    HbLazyTensor::StepMarker({}, nullptr, {}, true);
    HbLazyTensor::StepMarkerFinish();
  }
}
