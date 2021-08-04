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

class GenOps : public habana_lazy_test::LazyTest {
  const torch::IntArrayRef m_dims = torch::IntArrayRef({2, 3, 2, 3});
  std::vector<torch::Tensor> m_inputs;
  std::vector<torch::Tensor> m_hinputs;

  void Compare(
      const torch::Tensor& cpu_result,
      const torch::Tensor& habana_result,
      double rtol = 1e-03,
      double atol = 1e-03) const {
    EXPECT_TRUE(habana_result.is_habana());
    torch::Tensor habana_result_on_cpu = habana_result.cpu();

    if (c10::isIntegralType(cpu_result.scalar_type(), /*includeBool=*/true)) {
      EXPECT_TRUE(torch::equal(cpu_result, habana_result_on_cpu))
          << "seed=" << GetSeed() << std::endl
          << "exp=" << std::endl
          << cpu_result << std::endl
          << "actual=" << std::endl
          << habana_result_on_cpu << std::endl;
    } else {
      EXPECT_TRUE(
          torch::allclose(cpu_result, habana_result_on_cpu, rtol, atol, true))
          << "seed=" << GetSeed() << std::endl
          << "exp=" << std::endl
          << cpu_result << std::endl
          << "actual=" << std::endl
          << habana_result_on_cpu << std::endl;
    }
  }

  void GenerateInputs(int num_inputs, torch::ScalarType dtype = torch::kFloat) {
    SetSeed();
    m_inputs.resize(num_inputs);
    m_hinputs.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      m_inputs[i] = dtype == torch::kBool ? torch::randn(m_dims) > 0
                                          : torch::randn(m_dims).to(dtype);
      m_hinputs[i] = m_inputs[i].to("hpu");
    }
  }

  // Generate inputs with different dtypes/sizes per input
  void GenerateInputs(
      int num_inputs,
      torch::ArrayRef<torch::IntArrayRef> sizes,
      std::vector<torch::ScalarType> dtypes = {}) {
    SetSeed();
    ASSERT_EQ(num_inputs, sizes.size());
    if (dtypes.empty()) {
      dtypes.resize(num_inputs, torch::kFloat);
    }

    m_inputs.resize(num_inputs);
    m_hinputs.resize(num_inputs);

    for (int i = 0; i < num_inputs; ++i) {
      m_inputs[i] = dtypes[i] == torch::kBool
          ? torch::randn(sizes.at(i)) > 0
          : torch::randn(sizes.at(i)).to(dtypes[i]);
      m_hinputs[i] = m_inputs[i].to("hpu");
    }
  }

 public:
  void TestOut(
      const std::function<torch::Tensor(torch::Tensor, torch::Tensor&)>& fn,
      torch::ScalarType dtype = torch::kFloat,
      torch::ScalarType out_dtype = torch::kFloat) {
    GenerateInputs(1, dtype);

    auto out = torch::empty({0}, dtype);
    auto hout = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

    fn(m_inputs[0], out);
    fn(m_hinputs[0], hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor&)>& fn,
      torch::ScalarType dtype,
      torch::ScalarType out_dtype) {
    GenerateInputs(2, dtype);

    auto out = torch::empty({0}, out_dtype);
    auto hout =
        torch::empty({0}, torch::TensorOptions(out_dtype).device("hpu"));

    fn(m_inputs[0], m_inputs[1], out);
    fn(m_hinputs[0], m_hinputs[1], hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor&)>& fn,
      torch::ScalarType dtype = torch::kFloat) {
    TestOut(fn, dtype, dtype);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Tensor, torch::Scalar, torch::Tensor&)>& fn,
      torch::ScalarType dtype,
      torch::ScalarType out_dtype) {
    GenerateInputs(1, dtype);

    auto out = torch::empty({0}, out_dtype);
    auto hout =
        torch::empty({0}, torch::TensorOptions(out_dtype).device("hpu"));

    torch::Scalar s = 1;

    fn(m_inputs[0], s, out);
    fn(m_hinputs[0], s, hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Tensor, torch::Scalar, torch::Tensor&)>& fn,
      torch::ScalarType dtype) {
    TestOut(fn, dtype, dtype);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Scalar, torch::Tensor, torch::Tensor&)>& fn,
      torch::ScalarType dtype,
      torch::ScalarType out_dtype) {
    GenerateInputs(1, dtype);

    auto out = torch::empty_like(m_inputs[0]);
    auto hout = torch::empty_like(m_hinputs[0]);

    torch::Scalar s = 1.1;

    fn(s, m_inputs[0], out);
    fn(s, m_hinputs[0], hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Scalar, torch::Tensor, torch::Tensor&)>& fn,
      torch::ScalarType dtype = torch::kFloat) {
    TestOut(fn, dtype, dtype);
  }

  void TestOut(
      const std::function<torch::Tensor(
          torch::Tensor,
          torch::Scalar,
          torch::Scalar,
          torch::Tensor&)>& fn,
      torch::ScalarType dtype,
      torch::ScalarType out_dtype) {
    GenerateInputs(1, dtype);

    auto out = torch::empty({0}, out_dtype);
    auto hout =
        torch::empty({0}, torch::TensorOptions(out_dtype).device("hpu"));

    torch::Scalar s1 = -0.05;
    torch::Scalar s2 = 0.05;

    fn(m_inputs[0], s1, s2, out);
    fn(m_hinputs[0], s1, s2, hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<torch::Tensor(
          torch::Tensor,
          torch::Scalar,
          torch::Scalar,
          torch::Tensor&)>& fn,
      torch::ScalarType dtype = torch::kFloat) {
    TestOut(fn, dtype, dtype);
  }

  void TestOut(
      const std::function<torch::Tensor(
          torch::Tensor,
          torch::Scalar,
          torch::Scalar,
          torch::Scalar,
          torch::Tensor&)>& fn,
      double s1 = 0.001,
      double s2 = 0.001,
      double s3 = 0.001) {
    GenerateInputs(1);

    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(m_inputs[0], s1, s2, s3, out);
    fn(m_hinputs[0], s1, s2, s3, hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Tensor, torch::Tensor, int64_t, torch::Tensor&)>&
          fn,
      int int_val = 0.01) {
    GenerateInputs(2);

    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(m_inputs[0], m_inputs[1], int_val, out);
    fn(m_hinputs[0], m_hinputs[1], int_val, hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<torch::Tensor(
          torch::Tensor,
          torch::Tensor,
          torch::Tensor,
          int64_t,
          torch::Tensor&)>& fn,
      int int_val = 1) {
    GenerateInputs(3);

    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(m_inputs[0], m_inputs[1], m_inputs[2], int_val, out);
    fn(m_hinputs[0], m_hinputs[1], m_hinputs[2], int_val, hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<torch::Tensor(
          torch::Tensor,
          int64_t,
          torch::optional<torch::ScalarType>,
          torch::Tensor&)>& fn,
      int int_val = -1) {
    GenerateInputs(1);

    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(m_inputs[0], int_val, torch::nullopt, out);
    fn(m_hinputs[0], int_val, torch::nullopt, hout);

    Compare(out, hout);
  }

  void TestOut(const std::function<
               torch::Tensor(torch::Tensor, int64_t, torch::Tensor&)>& fn) {
    GenerateInputs(1);
    int int_val = 2;
    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(m_inputs[0], int_val, out);
    fn(m_hinputs[0], int_val, hout);

    Compare(out, hout);
  }

  void TestInplace(const std::function<torch::Tensor&(torch::Tensor&)>& fn) {
    GenerateInputs(1);

    auto res = fn(m_inputs[0]);
    auto hres = fn(m_hinputs[0]);

    EXPECT_EQ(hres.storage().data_ptr(), m_hinputs[0].storage().data_ptr());
    Compare(res, hres);
  }

  void TestInplace(
      const std::function<torch::Tensor&(torch::Tensor&, torch::Scalar)>& fn) {
    GenerateInputs(1);
    torch::Scalar s = 0.001;

    auto res = fn(m_inputs[0], s);
    auto hres = fn(m_hinputs[0], s);

    EXPECT_EQ(hres.storage().data_ptr(), m_hinputs[0].storage().data_ptr());
    Compare(res, hres);
  }

  void TestFn(const std::function<torch::Tensor(torch::Tensor)>& fn) {
    GenerateInputs(1);

    auto res = fn(m_inputs[0]);
    auto hres = fn(m_hinputs[0]);

    Compare(res, hres);
  }

  void TestFn(
      const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& fn) {
    GenerateInputs(2);

    auto res = fn(m_inputs[0], m_inputs[1]);
    auto hres = fn(m_hinputs[0], m_hinputs[1]);

    Compare(res, hres);
  }

  void TestFn(const std::function<torch::Tensor(
                  torch::Tensor,
                  int64_t,
                  torch::optional<torch::ScalarType>)>& fn) {
    GenerateInputs(1);
    int64_t int_val = 2;

    auto res = fn(m_inputs[0], int_val, torch::nullopt);
    auto hres = fn(m_hinputs[0], int_val, torch::nullopt);

    Compare(res, hres);
  }

  void TestFn(const std::function<torch::Tensor(torch::Tensor, int64_t)>& fn) {
    GenerateInputs(1);
    int64_t int_val = -1;
    auto res = fn(m_inputs[0], int_val);
    auto hres = fn(m_hinputs[0], int_val);

    Compare(res, hres);
  }

  void TestFn(const std::function<
              torch::Tensor(torch::Tensor, torch::Scalar, torch::Scalar)>& fn) {
    GenerateInputs(1);
    torch::Scalar s1 = -1;
    torch::Scalar s2 = 1;
    auto res = fn(m_inputs[0], s1, s2);
    auto hres = fn(m_hinputs[0], s1, s2);

    Compare(res, hres);
  }

  void TestFn(const std::function<
              torch::Tensor(torch::Tensor, torch::Tensor, torch::Scalar)>& fn) {
    GenerateInputs(2);
    torch::Scalar s1 = -1.042;
    auto res = fn(m_inputs[0], m_inputs[1], s1);
    auto hres = fn(m_hinputs[0], m_hinputs[1], s1);

    Compare(res, hres);
  }

  void TestFnCustomSizes(
      const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& fn,
      torch::ArrayRef<torch::IntArrayRef> sizes) {
    GenerateInputs(2, sizes);
    auto res = fn(m_inputs[0], m_inputs[1]);
    auto hres = fn(m_hinputs[0], m_hinputs[1]);

    Compare(res, hres);
  }
};

TEST_F(GenOps, Fns) {
  // clang-format off
  TestFnCustomSizes(torch::prelu, {{3, 4, 4, 1}, {1, 4, 1, 1}});
  TestFn(static_cast<torch::Tensor (*)(const torch::Tensor&, const torch::Tensor&, torch::Scalar)>(torch::rsub));
  TestFn(static_cast<torch::Tensor (*)(const torch::Tensor&, int64_t)>(torch::logcumsumexp));
  TestFn(static_cast<torch::Tensor (*)(const torch::Tensor&, int64_t, torch::optional<torch::ScalarType>)>(torch::cumprod));
  TestFn(torch::ceil);
  TestFn(torch::cos);
  TestFn(torch::_cumprod);
  TestFn(torch::_cumsum);
  TestFn(torch::exp2);
  TestInplace(torch::asin_);
  TestInplace(torch::ceil_);
  TestInplace(torch::clamp_min_);
  TestInplace(torch::neg_);
  TestInplace(torch::sigmoid_);
  TestInplace(torch::sin_);
  TestInplace(torch::sqrt_);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::bitwise_and_outf), torch::kByte);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::bitwise_or_outf), torch::kShort);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::bitwise_xor_outf), torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::eq_outf), torch::kI32, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::ge_outf), torch::kBFloat16, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::gt_outf), torch::kI32, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::le_outf), torch::kBool, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(torch::lt_outf), torch::kFloat, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, int64_t, torch::optional<torch::ScalarType>, torch::Tensor&)>(torch::cumprod_outf));
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, int64_t, torch::optional<torch::ScalarType>, torch::Tensor&)>(torch::cumsum_outf));
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::bitwise_and_outf), torch::kInt64);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::bitwise_xor_outf), torch::kInt);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::eq_outf), torch::kChar, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::ge_outf), torch::kByte, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::gt_outf), torch::kFloat, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::le_outf), torch::kInt, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::lt_outf), torch::kInt, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Scalar, torch::Tensor&)>(torch::pow_outf));
  TestOut(static_cast<torch::Tensor& (*)(torch::Scalar, const torch::Tensor&, torch::Tensor&)>(torch::pow_outf));
  TestOut(torch::abs_outf);
  TestOut(torch::acosh_outf);
  TestOut(torch::acos_outf);
  TestOut(torch::add_outf);
  TestOut(torch::asinh_outf);
  TestOut(torch::asin_outf);
  TestOut(torch::atanh_outf);
  TestOut(torch::atan_outf);
  TestOut(torch::bitwise_not_outf, torch::kChar);
  TestOut(torch::ceil_outf);
  TestOut(torch::clamp_max_outf, torch::kLong);
  TestOut(torch::clamp_min_outf);
  TestOut(torch::clamp_outf, torch::kInt);
  TestOut(torch::cosh_outf);
  TestOut(torch::cos_outf);
  TestOut(torch::_cumprod_outf);
  TestOut(torch::_cumsum_outf);
  TestOut(torch::elu_outf, /*alpha*/0.001, /*scale*/1, /*input_scale*/1);
  TestOut(torch::erf_outf);
  TestOut(torch::exp2_outf);
  TestOut(torch::exp_outf);
  TestOut(torch::floor_outf);
  TestOut(torch::hardsigmoid_outf);
  TestOut(torch::leaky_relu_outf);
  TestOut(torch::log2_outf);
  TestOut(torch::log_outf);
  TestOut(torch::maximum_outf);
  TestOut(torch::minimum_outf);
  TestOut(torch::mse_loss_backward_outf, torch::Reduction::Reduction::None);
  TestOut(torch::mse_loss_outf, torch::Reduction::Reduction::None);
  TestOut(torch::neg_outf);
  TestOut(torch::reciprocal_outf);
  TestOut(torch::round_outf);
  TestOut(torch::rsqrt_outf);
  TestOut(torch::sgn_outf);
  TestOut(torch::sigmoid_backward_outf);
  TestOut(torch::sigmoid_outf);
  TestOut(torch::sign_outf);
  TestOut(torch::sinh_outf);
  TestOut(torch::sin_outf);
  TestOut(torch::sqrt_outf);
  TestOut(torch::sub_outf);
  TestOut(torch::tanh_backward_outf);
  TestOut(torch::tanh_outf);
  TestOut(torch::tan_outf);
  // clang-format on
}
