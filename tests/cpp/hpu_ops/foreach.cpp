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

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

#define FOREACH(outplace_op)                                                 \
  TEST_F(HpuOpTest, outplace_op) {                                           \
    static constexpr int n = 5;                                              \
    std::vector<at::Tensor> cpu_in;                                          \
    std::vector<at::Tensor> hpu_in;                                          \
    GenerateInputs(n, {{4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}}); \
    for (int i = 0; i < n; ++i) {                                            \
      cpu_in.push_back(GetCpuInput(i));                                      \
      hpu_in.push_back(GetHpuInput(i));                                      \
    }                                                                        \
                                                                             \
    auto exp = outplace_op(cpu_in);                                          \
    auto res = outplace_op(hpu_in);                                          \
                                                                             \
    for (int i = 0; i < n; ++i) {                                            \
      Compare(exp[i], res[i]);                                               \
    }                                                                        \
  }

#define FOREACH_(inplace_op)                                                 \
  TEST_F(HpuOpTest, inplace_op##_) {                                         \
    static constexpr int n = 5;                                              \
    std::vector<at::Tensor> cpu_in;                                          \
    std::vector<at::Tensor> hpu_in;                                          \
    GenerateInputs(n, {{4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}}); \
    for (int i = 0; i < n; ++i) {                                            \
      cpu_in.push_back(GetCpuInput(i));                                      \
      hpu_in.push_back(GetHpuInput(i));                                      \
    }                                                                        \
                                                                             \
    inplace_op##_(cpu_in);                                                   \
    inplace_op##_(hpu_in);                                                   \
                                                                             \
    for (int i = 0; i < n; ++i) {                                            \
      Compare(cpu_in[i], hpu_in[i]);                                         \
    }                                                                        \
  }

#define FOREACH_TESTS(op) FOREACH(_foreach_##op) FOREACH_(_foreach_##op)

#define FOREACH_SCALAR(inplace_op)                                           \
  TEST_F(HpuOpTest, inplace_op##_) {                                         \
    static constexpr int n = 5;                                              \
    std::vector<at::Tensor> cpu_in;                                          \
    std::vector<at::Tensor> hpu_in;                                          \
    GenerateInputs(n, {{4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}}); \
    for (int i = 0; i < n; ++i) {                                            \
      cpu_in.push_back(GetCpuInput(i));                                      \
      hpu_in.push_back(GetHpuInput(i));                                      \
    }                                                                        \
                                                                             \
    inplace_op##_(cpu_in);                                                   \
    inplace_op##_(hpu_in);                                                   \
                                                                             \
    for (int i = 0; i < n; ++i) {                                            \
      Compare(cpu_in[i], hpu_in[i]);                                         \
    }                                                                        \
  }

FOREACH_(_foreach_zero)
FOREACH_TESTS(exp)
FOREACH_TESTS(sqrt)
FOREACH_TESTS(abs)
FOREACH_TESTS(acos)
FOREACH_TESTS(asin)
FOREACH_TESTS(atan)
FOREACH_TESTS(ceil)
FOREACH_TESTS(cos)
FOREACH_TESTS(cosh)
FOREACH_TESTS(erf)
FOREACH_TESTS(erfc)
FOREACH_TESTS(expm1)
FOREACH_TESTS(floor)
FOREACH_TESTS(log)
FOREACH_TESTS(log10)
FOREACH_TESTS(log1p)
FOREACH_TESTS(log2)
FOREACH_TESTS(neg)
FOREACH_TESTS(tan)
FOREACH_TESTS(tanh)
FOREACH_TESTS(sin)
FOREACH_TESTS(sinh)
FOREACH_TESTS(round)
/*FOREACH_TESTS(lgamma)*/
FOREACH_TESTS(frac)
FOREACH_TESTS(reciprocal)
FOREACH_TESTS(sigmoid)
FOREACH_TESTS(trunc)

TEST_F(HpuOpTest, ForeachRound) {
  std::vector<at::Tensor> cpu_in;
  std::vector<at::Tensor> hpu_in;
  auto t1_cpu = torch::tensor({1.5, 2.5, 3.5, 4.5}, torch::kFloat);
  cpu_in.push_back(t1_cpu);
  auto cpu_out = torch::_foreach_round(cpu_in);

  auto t1_hpu = t1_cpu.to(torch::kHPU);
  hpu_in.push_back(t1_hpu);
  auto hpu_out = torch::_foreach_round(hpu_in);

  auto t2_cpu = cpu_out[0];
  auto t2_hpu = hpu_out[0];

  EXPECT_EQ(allclose(t2_hpu.to(torch::kCPU), t2_cpu, 0.001, 0.001), true);
  EXPECT_EQ(t2_hpu.dtype() == t2_cpu.dtype(), true);
}
