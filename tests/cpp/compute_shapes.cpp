/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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

TEST_F(ComputeShapes, sigmoid) {
  auto cpu_in = torch::randn({2, 3, 4, 5});
  auto hpu_in = cpu_in.to("hpu");

  EXPECT_TRUE(at::allclose(at::sigmoid(cpu_in), at::sigmoid(hpu_in).cpu()));
}

// Disabled due to SW-92670
TEST_F(ComputeShapes, DISABLED_ge) {
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
