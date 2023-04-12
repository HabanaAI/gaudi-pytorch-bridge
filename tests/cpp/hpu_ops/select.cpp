/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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