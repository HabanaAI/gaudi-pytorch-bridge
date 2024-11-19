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
#include "backend/random.h"

using namespace habana_lazy;
using namespace at;

class LazyRandomGenKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyRandomGenKernelTest, RandpermOutTest) {
  constexpr int n = 10;

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);

  torch::manual_seed(0);
  habana::detail::getDefaultHPUGenerator().set_current_seed(0);
  auto eager = torch::randperm(n, hb_options);
  auto eager_cpu = eager.to(torch::kCPU);

  SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 1, 0);

  torch::manual_seed(0);
  habana::detail::getDefaultHPUGenerator().set_current_seed(0);
  auto lazy = torch::randperm(n, hb_options);
  auto lazy_cpu = lazy.to(torch::kCPU);

  auto equal = eager_cpu.equal(lazy_cpu);
  EXPECT_EQ(equal, true);

  UNSET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE);
}
