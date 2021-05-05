/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/torch.h>

class FallbackTest : public ::testing::TestWithParam<bool>,
                     public habana_lazy_test::EnvHelper {
  void SetUp() override {
    auto isLazy = GetParam();
    if (isLazy) {
      SetLazyMode(); // Lazy=1 mode
    } else {
      SetEagerMode(); // Eager mode
    }
  }
};

struct PrintToStringParamName {
  template <class ParamType>
  std::string operator()(
      const ::testing::TestParamInfo<ParamType>& info) const {
    auto isLazy = static_cast<bool>(info.param);
    return isLazy ? "lazy" : "eager";
  }
};
INSTANTIATE_TEST_SUITE_P(
    sanity,
    FallbackTest,
    ::testing::Bool(),
    PrintToStringParamName());

TEST_P(FallbackTest, Simple) {
  auto ones = torch::ones(10, "hpu");
  auto res = ones.digamma();
  res = res.add(ones);

  constexpr float ones_digamma = 0.42278409;
  auto exp = torch::full(10, ones_digamma);
  EXPECT_TRUE(allclose(exp, res.to("cpu")));
}

TEST_P(FallbackTest, Inplace) {
  auto randt = torch::rand(10).to("hpu");
  int p = 2; // Must be greater than 1

  // All element of t must be greater than (p-1)/2
  const float delta = 0.1;
  float minValueOfElement = (p - 1) / 2.0 + delta;
  auto t = randt + (1.0 + minValueOfElement);

  auto res = t.mvlgamma_(p);
  EXPECT_EQ(t.storage().data_ptr().get(), res.storage().data_ptr().get());
}

TEST_P(FallbackTest, NonSupportedAsStrided) {
  torch::Tensor A = torch::rand({3, 3, 3, 3, 3});
  torch::Tensor hA = A.to(torch::kHABANA);
  at::Tensor Out = A.as_strided({2, 2}, {1, 2});
  at::Tensor hOut = hA.as_strided({2, 2}, {1, 2});

  Out.div_(4);
  hOut.div_(4);

  EXPECT_TRUE(allclose(Out, hOut.to("cpu"))) << Out << hOut.to("cpu");
}

TEST_P(FallbackTest, bitwise_xor) {
  auto self = torch::tensor({-1, -2, 3}, torch::kInt8);
  auto other = torch::tensor({1, 0, 3}, torch::kInt8);
  at::Tensor Out = self.bitwise_xor(other);

  auto hself = self.to(torch::kHABANA);
  auto hother = other.to(torch::kHABANA);
  // Uses out variant - bitwise_or_out internally
  at::Tensor hOut = hself.bitwise_xor(hother);

  EXPECT_TRUE(allclose(Out, hOut.to("cpu"))) << Out << hOut.to("cpu");
}
