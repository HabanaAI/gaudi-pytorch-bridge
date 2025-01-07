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
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class LazyPoolKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyPoolKernelTest, MaxPoolBWDTest) {
  auto maxpool = []() {
    auto input_tensor =
        torch::arange(20, torch::dtype(torch::kFloat).requires_grad(true))
            .reshape({1, 1, 4, 5}); // nchw
    auto cpu_pool = torch::max_pool2d(input_tensor, 3, 1);
    auto cpu_out = torch::relu(cpu_pool);

    // fwd propga
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    auto outHabana1 = torch::max_pool2d_with_indices(
        tHabanaX, {3, 3}, {1, 1}, {0, 0}, {1, 1}, true);
    torch::Tensor outHabana = torch::relu(std::get<0>(outHabana1));

    // bwd propga with dummy grad tensor
    auto grad_tensor =
        torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
            .reshape({1, 1, 2, 3});
    torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
    outHabana.backward({tHabanaG}, false, true);

    auto out_cpu_lazy = outHabana.to(torch::kCPU);
    ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
  };
  maxpool();
  // call with shape validation enabled
  setenv("PT_HPU_VALIDATE_COMPUTE_SHAPE", "true", 1);
  maxpool();
  unsetenv("PT_HPU_VALIDATE_COMPUTE_SHAPE");
}

TEST_F(LazyPoolKernelTest, AvgPoolTest) {
  auto input_tensor =
      torch::arange(20, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 4, 5}); // nchw
  auto cpu_out = torch::avg_pool2d(input_tensor, 3, 1);

  // fwd propagation
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  auto outHabana =
      torch::avg_pool2d(tHabanaX, {3, 3}, {1, 1}, {0, 0}, false, true);

  ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), cpu_out));

  // bwd propagation with dummy grad tensor
  auto grad_tensor =
      torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 3});
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHPU);
  outHabana.backward({tHabanaG}, false, true);

  auto out_cpu_lazy = outHabana.to(torch::kCPU);

  ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
}
