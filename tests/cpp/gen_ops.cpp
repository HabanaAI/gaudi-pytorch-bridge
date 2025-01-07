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
#include "hpu_ops/util.h"

class GenOps : public HpuOpTestUtil {
 public:
  void TestOut(
      const std::function<torch::Tensor(torch::Tensor, torch::Tensor&)>& fn,
      torch::ScalarType dtype = torch::kFloat,
      torch::ScalarType out_dtype = torch::kFloat) {
    GenerateInputs(1, dtype);

    auto out = torch::empty({0}, dtype);
    auto hout = torch::empty({0}, torch::TensorOptions(dtype).device("hpu"));

    fn(GetCpuInput(0), out);
    fn(GetHpuInput(0), hout);

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

    fn(GetCpuInput(0), GetCpuInput(1), out);
    fn(GetHpuInput(0), GetHpuInput(1), hout);

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

    fn(GetCpuInput(0), s, out);
    fn(GetHpuInput(0), s, hout);

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

    auto out = torch::empty_like(GetCpuInput(0));
    auto hout = torch::empty_like(GetHpuInput(0));

    torch::Scalar s = 1.1;

    fn(s, GetCpuInput(0), out);
    fn(s, GetHpuInput(0), hout);

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

    fn(GetCpuInput(0), s1, s2, out);
    fn(GetHpuInput(0), s1, s2, hout);

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

    fn(GetCpuInput(0), s1, s2, s3, out);
    fn(GetHpuInput(0), s1, s2, s3, hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<
          torch::Tensor(torch::Tensor, torch::Tensor, int64_t, torch::Tensor&)>&
          fn,
      int int_val = 2) {
    GenerateInputs(2);

    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(GetCpuInput(0), GetCpuInput(1), int_val, out);
    fn(GetHpuInput(0), GetHpuInput(1), int_val, hout);

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

    fn(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), int_val, out);
    fn(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), int_val, hout);

    Compare(out, hout);
  }

  void TestOut(
      const std::function<torch::Tensor(
          torch::Tensor,
          int64_t,
          torch::optional<torch::ScalarType>,
          torch::Tensor&)>& fn,
      int int_val = -1) {
    GenerateInputs(1, torch::kInt);

    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(GetCpuInput(0), int_val, torch::kFloat, out);
    fn(GetHpuInput(0), int_val, torch::kFloat, hout);

    Compare(out, hout);
  }

  void TestOut(const std::function<
               torch::Tensor(torch::Tensor, int64_t, torch::Tensor&)>& fn) {
    GenerateInputs(1);
    int int_val = 2;
    auto out = torch::empty({0});
    auto hout = torch::empty({0}, "hpu");

    fn(GetCpuInput(0), int_val, out);
    fn(GetHpuInput(0), int_val, hout);

    Compare(out, hout);
  }

  void TestOut(const std::function<torch::Tensor&(
                   const torch::Tensor&,
                   const torch::optional<torch::Tensor>&,
                   const torch::optional<torch::Tensor>&,
                   torch::Tensor&)>& fn) {
    GenerateInputs(3);

    auto out = torch::empty_like(GetCpuInput(0));
    auto hout = torch::empty_like(GetHpuInput(0));

    fn(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), out);
    fn(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), hout);

    Compare(out, hout);
  }

  void TestInplace(const std::function<torch::Tensor&(torch::Tensor&)>& fn) {
    GenerateInputs(1);

    auto res = fn(GetCpuInput(0));
    auto hres = fn(GetHpuInput(0));

    EXPECT_EQ(hres.storage().data_ptr(), GetHpuInput(0).storage().data_ptr());
    Compare(res, hres);
  }

  void TestInplace(
      const std::function<torch::Tensor&(torch::Tensor&, const torch::Scalar&)>&
          fn) {
    GenerateInputs(1);
    torch::Scalar s = 0.001;

    auto res = fn(GetCpuInput(0), s);
    auto hres = fn(GetHpuInput(0), s);

    EXPECT_EQ(hres.storage().data_ptr(), GetHpuInput(0).storage().data_ptr());
    Compare(res, hres);
  }

  void TestInplace(
      const std::function<torch::Tensor&(torch::Tensor&, const torch::Tensor&)>&
          fn) {
    GenerateInputs(2);

    auto res = fn(GetCpuInput(0), GetCpuInput(1));
    auto hres = fn(GetHpuInput(0), GetHpuInput(1));

    EXPECT_EQ(hres.storage().data_ptr(), GetHpuInput(0).storage().data_ptr());
    Compare(res, hres);
  }

  void TestInplace(const std::function<torch::Tensor&(
                       torch::Tensor&,
                       const torch::optional<torch::Tensor>&,
                       const torch::optional<torch::Tensor>&)>& fn) {
    GenerateInputs(3);

    auto res = fn(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
    auto hres = fn(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));

    EXPECT_EQ(hres.storage().data_ptr(), GetHpuInput(0).storage().data_ptr());
    Compare(res, hres);
  }

  void TestInplace(const std::function<torch::Tensor&(
                       torch::Tensor&,
                       const torch::Scalar&,
                       const torch::Scalar&)>& fn) {
    GenerateInputs(1);
    torch::Scalar s1 = -0.05;
    torch::Scalar s2 = 0.05;

    auto res = fn(GetCpuInput(0), s1, s2);
    auto hres = fn(GetHpuInput(0), s1, s2);

    EXPECT_EQ(hres.storage().data_ptr(), GetHpuInput(0).storage().data_ptr());
    Compare(res, hres);
  }

  void TestFn(const std::function<torch::Tensor(torch::Tensor)>& fn) {
    GenerateInputs(1);

    auto res = fn(GetCpuInput(0));
    auto hres = fn(GetHpuInput(0));

    Compare(res, hres);
  }

  void TestFn(
      const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& fn) {
    GenerateInputs(2);

    auto res = fn(GetCpuInput(0), GetCpuInput(1));
    auto hres = fn(GetHpuInput(0), GetHpuInput(1));

    Compare(res, hres);
  }

  void TestFn(const std::function<torch::Tensor(
                  torch::Tensor,
                  int64_t,
                  torch::optional<torch::ScalarType>)>& fn) {
    GenerateInputs(1, torch::kInt);
    int64_t int_val = 2;

    auto res = fn(GetCpuInput(0), int_val, torch::kFloat);
    auto hres = fn(GetHpuInput(0), int_val, torch::kFloat);

    Compare(res, hres);
  }

  void TestFn(const std::function<torch::Tensor(torch::Tensor, int64_t)>& fn) {
    GenerateInputs(1);
    int64_t int_val = -1;
    auto res = fn(GetCpuInput(0), int_val);
    auto hres = fn(GetHpuInput(0), int_val);

    Compare(res, hres);
  }

  void TestFn(const std::function<
              torch::Tensor(torch::Tensor, torch::Scalar, torch::Scalar)>& fn) {
    GenerateInputs(1);
    torch::Scalar s1 = -1;
    torch::Scalar s2 = 1;
    auto res = fn(GetCpuInput(0), s1, s2);
    auto hres = fn(GetHpuInput(0), s1, s2);

    Compare(res, hres);
  }

  void TestFn(const std::function<
              torch::Tensor(torch::Tensor, torch::Tensor, torch::Scalar)>& fn) {
    GenerateInputs(2);
    torch::Scalar s1 = -1.042;
    auto res = fn(GetCpuInput(0), GetCpuInput(1), s1);
    auto hres = fn(GetHpuInput(0), GetHpuInput(1), s1);

    Compare(res, hres);
  }

  void TestFn(const std::function<torch::Tensor(
                  const torch::Tensor&,
                  const torch::optional<torch::Tensor>&,
                  const torch::optional<torch::Tensor>&)>& fn) {
    GenerateInputs(3);

    auto res = fn(GetCpuInput(0), GetCpuInput(1), GetCpuInput(2));
    auto hres = fn(GetHpuInput(0), GetHpuInput(1), GetHpuInput(2));

    Compare(res, hres);
  }

  void TestFnCustomSizes(
      const std::function<
          torch::Tensor(torch::Tensor, torch::Tensor, int64_t, int64_t, bool)>&
          fn,
      torch::ArrayRef<torch::IntArrayRef> sizes) {
    GenerateInputs(2, sizes);
    int64_t s1 = 0;
    int64_t s2 = 0;
    bool s3 = false;
    auto res = fn(GetCpuInput(0), GetCpuInput(1), s1, s2, s3);
    auto hres = fn(GetHpuInput(0), GetHpuInput(1), s1, s2, s3);

    Compare(res, hres);
  }

  void TestFnCustomSizes(
      const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& fn,
      torch::ArrayRef<torch::IntArrayRef> sizes) {
    GenerateInputs(2, sizes);
    auto res = fn(GetCpuInput(0), GetCpuInput(1));
    auto hres = fn(GetHpuInput(0), GetHpuInput(1));

    Compare(res, hres);
  }

  void TestOutClampCustom(const std::function<torch::Tensor&(
                              const torch::Tensor&,
                              const torch::optional<torch::Tensor>&,
                              const torch::optional<torch::Tensor>&,
                              torch::Tensor&)>& fn) {
    GenerateInputs(2);

    auto out = torch::empty_like(GetCpuInput(0));
    auto hout = torch::empty_like(GetHpuInput(0));

    // calling to clamp with Max tensor is not defined
    fn(GetCpuInput(0), GetCpuInput(1), torch::nullopt, out);
    fn(GetHpuInput(0), GetHpuInput(1), torch::nullopt, hout);

    Compare(out, hout);
  }
};

TEST_F(GenOps, Fns) {
  // clang-format off
  TestFnCustomSizes(torch::prelu, {{3, 4, 4, 1}, {4}});
  TestFnCustomSizes(torch::prelu, {{5, 7}, {7}});
  TestFnCustomSizes(static_cast<torch::Tensor (*)(const torch::Tensor&, const torch::Tensor&, int64_t, int64_t, bool)>(torch::grid_sampler_2d), {{2, 3, 4, 4}, {2, 3, 3, 2}});
  TestFn(static_cast<torch::Tensor (*)(const torch::Tensor&, const torch::Tensor&, const torch::Scalar&)>(torch::rsub));
  TestFn(static_cast<torch::Tensor (*)(const torch::Tensor&, int64_t)>(torch::logcumsumexp));
  TestFn(static_cast<torch::Tensor (*)(const torch::Tensor&, int64_t, torch::optional<torch::ScalarType>)>(torch::cumprod));
  TestFn(torch::ceil);
  TestFn(torch::cos);
  TestFn(torch::exp2);
  TestInplace(torch::asin_);
  TestInplace(torch::ceil_);
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
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::bitwise_and_outf), torch::kInt64);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::bitwise_xor_outf), torch::kInt);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::eq_outf), torch::kChar, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::ge_outf), torch::kByte, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::gt_outf), torch::kFloat, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::le_outf), torch::kInt, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::lt_outf), torch::kInt, torch::kBool);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::pow_outf));
  TestOut(torch::abs_outf);
  TestOut(torch::acosh_outf);
  TestOut(torch::acos_outf);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::add_outf));
  TestOut(torch::asinh_outf);
  TestOut(torch::asin_outf);
  TestOut(torch::atanh_outf);
  TestOut(torch::atan_outf);
  TestOut(torch::bitwise_not_outf, torch::kChar);
  TestOut(torch::ceil_outf);
  TestOut(torch::cosh_outf);
  TestOut(torch::cos_outf);
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
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, torch::Tensor&)>(torch::round_outf));
  TestOut(torch::rsqrt_outf);
  TestOut(torch::sgn_outf);
  TestOut(torch::sigmoid_backward_outf);
  TestOut(torch::sigmoid_outf);
  TestOut(torch::sign_outf);
  TestOut(torch::sinh_outf);
  TestOut(torch::sin_outf);
  TestOut(torch::sqrt_outf);
  TestOut(static_cast<torch::Tensor& (*)(const torch::Tensor&, const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(torch::sub_outf));
  TestOut(torch::tanh_backward_outf);
  TestOut(torch::tanh_outf);
  TestOut(torch::tan_outf);
  // clang-format on
}

TEST_F(GenOps, clampGenTensor) {
  TestFn(static_cast<torch::Tensor (*)(
             const torch::Tensor&,
             const torch::optional<torch::Tensor>&,
             const torch::optional<torch::Tensor>&)>(torch::clamp));
}

TEST_F(GenOps, clampGenScalar) {
  TestFn(static_cast<torch::Tensor (*)(
             const torch::Tensor&,
             const torch::optional<torch::Scalar>&,
             const torch::optional<torch::Scalar>&)>(torch::clamp));
}

TEST_F(GenOps, clampGenMinTensor) {
  TestFn(static_cast<torch::Tensor (*)(
             const torch::Tensor&, const torch::Tensor&)>(torch::clamp_min));
}

TEST_F(GenOps, clampGenMinScalar) {
  TestFn(static_cast<torch::Tensor (*)(
             const torch::Tensor&, const torch::Scalar&)>(torch::clamp_min));
}

TEST_F(GenOps, clampGenMaxTensor) {
  TestFn(static_cast<torch::Tensor (*)(
             const torch::Tensor&, const torch::Tensor&)>(torch::clamp_max));
}

TEST_F(GenOps, clampGenMaxScalar) {
  TestFn(static_cast<torch::Tensor (*)(
             const torch::Tensor&, const torch::Scalar&)>(torch::clamp_max));
}

TEST_F(GenOps, clampGenInplaceTensor) {
  TestInplace(
      static_cast<
          torch::
              Tensor& (*)(torch::Tensor&, const torch::optional<torch::Tensor>&, const torch::optional<torch::Tensor>&)>(
          torch::clamp_));
}

TEST_F(GenOps, clampGenInplaceScalar) {
  TestInplace(
      static_cast<
          torch::
              Tensor& (*)(torch::Tensor&, const torch::optional<torch::Scalar>&, const torch::optional<torch::Scalar>&)>(
          torch::clamp_));
}

TEST_F(GenOps, clampGenInplaceMinTensor) {
  TestInplace(
      static_cast<torch::Tensor& (*)(torch::Tensor&, const torch::Tensor&)>(
          torch::clamp_min_));
}

TEST_F(GenOps, clampGenInplaceMinScalar) {
  TestInplace(
      static_cast<torch::Tensor& (*)(torch::Tensor&, const torch::Scalar&)>(
          torch::clamp_min_));
}

TEST_F(GenOps, clampGenInplaceMaxTensor) {
  TestInplace(
      static_cast<torch::Tensor& (*)(torch::Tensor&, const torch::Tensor&)>(
          torch::clamp_max_));
}

TEST_F(GenOps, clampGenInplaceMaxScalar) {
  TestInplace(
      static_cast<torch::Tensor& (*)(torch::Tensor&, const torch::Scalar&)>(
          torch::clamp_max_));
}

TEST_F(GenOps, clampGenOutTensor) {
  TestOut(static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::optional<torch::Tensor>&, const torch::optional<torch::Tensor>&, torch::Tensor&)>(
      torch::clamp_outf));
}

TEST_F(GenOps, clampGenOutScalar) {
  TestOut(static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::optional<torch::Scalar>&, const torch::optional<torch::Scalar>&, torch::Tensor&)>(
      torch::clamp_outf));
}

TEST_F(GenOps, clampGenOutMinTensor) {
  TestOut(static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(
      torch::clamp_min_outf));
}

TEST_F(GenOps, clampGenOutMinScalar) {
  TestOut(static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(
      torch::clamp_min_outf));
}

TEST_F(GenOps, clampGenOutMaxTensor) {
  TestOut(static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(
      torch::clamp_max_outf));
}

TEST_F(GenOps, clampGenOutMaxScalar) {
  TestOut(static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(
      torch::clamp_max_outf));
}

TEST_F(GenOps, clampGenOutMaxTensorLong) {
  TestOut(
      static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&)>(
          torch::clamp_max_outf),
      torch::kLong);
}

TEST_F(GenOps, clampGenOutMaxScalarLong) {
  TestOut(
      static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::Scalar&, torch::Tensor&)>(
          torch::clamp_max_outf),
      torch::kLong);
}

TEST_F(GenOps, clamGenOutClamCustom) {
  TestOutClampCustom(
      static_cast<
          torch::
              Tensor& (*)(const torch::Tensor&, const torch::optional<torch::Tensor>&, const torch::optional<torch::Tensor>&, torch::Tensor&)>(
          torch::clamp_outf));
}
