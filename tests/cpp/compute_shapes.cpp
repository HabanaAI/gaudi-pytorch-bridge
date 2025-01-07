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

class ComputeShapes : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, 1, 1);
  }
  void TearDown() override {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, 0, 1);
  }
};

TEST_F(ComputeShapes, atan) {
  auto cpu_in = torch::randn({64, 42});
  auto hpu_in = cpu_in.to("hpu");

  EXPECT_TRUE(at::allclose(at::atan(cpu_in), at::atan(hpu_in).cpu()));
}

TEST_F(ComputeShapes, atan__CmptOpShp) {
  auto cpu_in = torch::randn({64, 42});
  auto hpu_in = cpu_in.to("hpu");

  torch::atan_(cpu_in);
  torch::atan_(hpu_in);

  EXPECT_TRUE(at::allclose(cpu_in, hpu_in.to("cpu")));
}

TEST_F(ComputeShapes, atanOut_CmptOpShp) {
  auto cpu_in = torch::randn({64, 42});
  auto hpu_in = cpu_in.to("hpu");
  auto outh = torch::empty_like(hpu_in);
  auto out = torch::empty_like(cpu_in);

  out = torch::atan_outf(cpu_in, out);
  outh = torch::atan_outf(hpu_in, outh);

  EXPECT_TRUE(at::allclose(out, outh.to("cpu")));
}

TEST_F(ComputeShapes, sigmoid) {
  auto cpu_in = torch::randn({2, 3, 4, 5});
  auto hpu_in = cpu_in.to("hpu");

  EXPECT_TRUE(at::allclose(at::sigmoid(cpu_in), at::sigmoid(hpu_in).cpu()));
}

TEST_F(ComputeShapes, ge) {
  auto cpu_in1 = torch::randn({42}).to(at::kBFloat16);
  auto cpu_in2 = torch::randn({2, 42});

  auto hpu_in1 = cpu_in1.to("hpu");
  auto hpu_in2 = cpu_in2.to("hpu");

  EXPECT_TRUE(
      at::allclose(at::ge(cpu_in1, cpu_in2), at::ge(hpu_in1, hpu_in2).cpu()));
}

TEST_F(ComputeShapes, bce) {
  auto cpu_in1 = torch::rand({10, 12});
  auto cpu_in2 = torch::rand({10, 12});

  auto hpu_in1 = cpu_in1.to("hpu");
  auto hpu_in2 = cpu_in2.to("hpu");

  EXPECT_TRUE(at::allclose(
      at::binary_cross_entropy(cpu_in1, cpu_in2),
      at::binary_cross_entropy(hpu_in1, hpu_in2).cpu()));
}

TEST_F(ComputeShapes, index_cmptopshp) {
  torch::Tensor input_cpu = torch::arange(4).reshape({2, 2});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{torch::tensor({{0, 1}, {0, 1}})};
  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  // auto tensorlist = indices.vec();
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list{};
  // auto tensorlist = indices.vec();
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_list);

  // TODO: Check index_hpu_lazy why this long cast is required
  bool equal =
      out_cpu.allclose(out_hpu.to(torch::kCPU).to(at::kLong), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}
