#include <gtest/gtest.h>
#include <torch/torch.h>

class FallbackTest : public ::testing::TestWithParam<bool> {
  void SetUp() override {
    auto isLazy = GetParam();
    if (isLazy) {
      setenv("PT_HPU_LAZY_MODE", "1", 1);
    }
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
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
  auto ones = torch::ones(10, "habana");
  auto res = ones.digamma();
  res = res.add(ones);

  constexpr float ones_digamma = 0.42278409;
  auto exp = torch::full(10, ones_digamma);
  EXPECT_TRUE(allclose(exp, res.to("cpu")));
}

TEST_P(FallbackTest, Inplace) {
  auto t = torch::rand(10).to("habana");
  auto res = t.acosh_();
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

TEST_P(FallbackTest, bitwise_or) {
  auto self = torch::tensor({-1, -2, 3}, torch::kInt8);
  auto other = torch::tensor({1, 0, 3}, torch::kInt8);
  at::Tensor Out = self.bitwise_or(other);

  auto hself = self.to(torch::kHABANA);
  auto hother = other.to(torch::kHABANA);
  // Uses out variant - bitwise_or_out internally
  at::Tensor hOut = hself.bitwise_or(hother);

  EXPECT_TRUE(allclose(Out, hOut.to("cpu"))) << Out << hOut.to("cpu");
}
