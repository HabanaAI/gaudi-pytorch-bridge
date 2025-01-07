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
#include "tests/cpp/habana_lazy_test_infra.h"
using namespace habana_lazy;
using namespace at;

class LazyMiscTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyMiscTest, CatchExceptionTest) {
  // TBD: revisit to check if there is anyway to enable this with DS
  bool org_state = habana_helpers::GetRefineDynamicShapeStatus();
  if (org_state) {
    habana_helpers::DisableRefineDynamicShape();
  }

  auto x = torch::randn({2, 3});
  auto y1 = torch::randn({4, 3});

  torch::Tensor hx = x.to(torch::kHPU);
  torch::Tensor hy1 = y1.to(torch::kHPU);

  try {
    auto z = torch::mm(hx, hy1).to(torch::kCPU);
  } catch (...) {
    auto y2 = torch::randn({3, 3});
    torch::Tensor hy2 = y2.to(torch::kHPU);
    auto z = torch::mm(hx, hy2).to(torch::kCPU);
    auto z_cpu = torch::mm(x, y2);
    EXPECT_EQ(allclose(z, z_cpu, 0.001, 0.001), true);
    return;
  }
  EXPECT_EQ(false, true);

  if (org_state) {
    habana_helpers::EnableRefineDynamicShape();
  }
}

TEST_F(LazyMiscTest, CloneTest) {
  auto x = torch::randn({2, 3});
  auto hx = x.to(torch::kHPU);
  x = torch::relu(x);
  auto hy = hx.clone();
  hy = torch::relu(hy);
  auto y = hy.to(torch::kCPU);

  EXPECT_EQ(allclose(x, y, 0.001, 0.001), true);
}

TEST_F(LazyMiscTest, CloneIRTest) {
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHPU);
  tensor_in1 = tensor_in1.relu();
  tensor_in1 = tensor_in1.clone();
  tensor_in1 = tensor_in1.relu();
  auto hl_result = SyncAndGetHbLazyTensor(tensor_in1);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
  auto str = po_data.post_order[0]->ToString();
  auto cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);

  std::vector<at::Tensor> input_list{tensor_in1};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("habana_d2d_memcpy", 0, true)
      ->run(*hlexec->get_graph());
}

TEST_F(LazyMiscTest, SliceInsertTest) {
  auto cpu_tensor = torch::randn({10, 20, 30});
  torch::Tensor tensor_in1 = cpu_tensor.to(torch::kHPU);
  auto cpu_ref = cpu_tensor.slice(1, 1, 9, 3);
  cpu_ref = cpu_ref.slice(2, 11, 29, 5);
  cpu_ref.add_(1);

  auto hpu = tensor_in1.slice(1, 1, 9, 3);
  hpu = hpu.slice(2, 11, 29, 5);
  hpu.add_(1);

  EXPECT_EQ(allclose(cpu_ref, hpu.to("cpu"), 0.001, 0.001), true);
}

TEST_F(LazyMiscTest, SliceInsertIRTest) {
  // TBD: revisit to check if there is anyway to enable this with DS
  bool org_state = habana_helpers::GetRefineDynamicShapeStatus();
  if (org_state) {
    habana_helpers::DisableRefineDynamicShape();
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SLICE_INSERT)) {
    torch::Tensor tensor_in1 = torch::randn({2, 10}).to(torch::kHPU);
    tensor_in1 = tensor_in1.slice(1, 1, 9, 3);
    tensor_in1 = tensor_in1.add_(1);
    auto tensor_in2 = tensor_in1.relu();

    auto hl_result = SyncAndGetHbLazyTensor(tensor_in2);
    std::vector<HbLazyTensor> tensors = {hl_result};
    std::vector<int> indices = {0};

    auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
    std::vector<at::Tensor> input_list{tensor_in1, tensor_in2};

    auto stack = torch::jit::Stack(
        std::make_move_iterator(input_list.begin()),
        std::make_move_iterator(input_list.end()));

    exec::HlExec* hlexec = new exec::HlExec();
    hlexec->GetOrCreate(po_data, stack);

    torch::jit::testing::FileCheck()
        .check_count("hpu::slice_insert", 1)
        ->run(*hlexec->get_graph());
  }

  if (org_state) {
    habana_helpers::EnableRefineDynamicShape();
  }
}
