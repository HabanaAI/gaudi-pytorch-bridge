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

class HpuSelectOpTest : public HpuOpTestUtil {};

TEST_F(HpuSelectOpTest, SelectNDimsTest) {
  torch::Tensor a =
      torch::randn({2, 3, 4, 5, 6, 4}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 2;

  auto h_out = torch::select(h_a, dim, 3);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select(a, dim, 3);

  EXPECT_TRUE(allclose(h_cout, cout, 0, 0));
}