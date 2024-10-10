/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, index) {
  torch::Tensor input_cpu = torch::arange(4).reshape({2, 2});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{torch::tensor({{0, 1}, {1, 1}})};

  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_list);

  EXPECT_TRUE(torch::equal(out_cpu, out_hpu.cpu()));
}
