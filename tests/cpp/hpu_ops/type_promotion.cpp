/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "util.h"

const std::array DtypesList{
    torch::kDouble,
    torch::kFloat,
    torch::kBFloat16,
    torch::kLong,
    torch::kInt,
    torch::kShort,
    torch::kByte,
    torch::kChar,
    torch::kBool,
};

const std::array DefaultDtypesList{
    torch::kFloat,
    torch::kBFloat16,
    torch::kDouble};

struct PrintParam1 {
  template <class ParamType>
  std::string operator()(const ::testing::TestParamInfo<ParamType>& p) const {
    std::ostringstream os;
    os << static_cast<c10::ScalarType>(std::get<0>(p.param));
    return os.str();
  }
};

struct PrintParam2 {
  template <class ParamType>
  std::string operator()(const ::testing::TestParamInfo<ParamType>& p) const {
    std::ostringstream os;
    os << static_cast<c10::ScalarType>(std::get<0>(p.param));
    os << "x";
    os << static_cast<c10::ScalarType>(std::get<1>(p.param));
    return os.str();
  }
};

struct PrintParam3 {
  template <class ParamType>
  std::string operator()(const ::testing::TestParamInfo<ParamType>& p) const {
    std::ostringstream os;
    os << static_cast<c10::ScalarType>(std::get<0>(p.param));
    os << "x";
    os << static_cast<c10::ScalarType>(std::get<1>(p.param));
    os << "x";
    os << static_cast<c10::ScalarType>(std::get<2>(p.param));
    return os.str();
  }
};

class BinaryTypePromotion : public HpuOpTestUtil,
                            public testing::WithParamInterface<
                                std::tuple<c10::ScalarType, c10::ScalarType>> {
};

TEST_P(BinaryTypePromotion, mul) {
  const auto& testParams = GetParam();
  const auto dtype0 = std::get<0>(testParams);
  const auto dtype1 = std::get<1>(testParams);
  GenerateInputs(2, {dtype0, dtype1});

  auto exp = torch::mul(GetCpuInput(0), GetCpuInput(1));
  auto res = torch::mul(GetHpuInput(0), GetHpuInput(1));

  switch (exp.scalar_type()) {
    case at::kByte:
    case at::kChar:
      // HPU results saturate while CPU results wrap around when overflow
      // occurs, so skip value comparison but check the scalar type of the
      // output and run the operation.

      EXPECT_EQ(exp.scalar_type(), res.scalar_type());
      res.cpu();
      return;
    default:
      break;
  }
  Compare(exp, res);
}

INSTANTIATE_TEST_SUITE_P(
    TypePromotion,
    BinaryTypePromotion,
    ::testing::Combine(
        ::testing::ValuesIn(DtypesList),
        ::testing::ValuesIn(DtypesList)),
    PrintParam2());

/////////////////////////////////////////////////////////////////////////////////////

class BinaryIntToFloatPromotion
    : public HpuOpTestUtil,
      public testing::WithParamInterface<
          std::tuple<c10::ScalarType, c10::ScalarType, c10::ScalarType>> {
  c10::ScalarType old_dtype_ = c10::ScalarType::Undefined;
  void SetUp() override {
    old_dtype_ = torch::get_default_dtype_as_scalartype();
  }
  void TearDown() override {
    torch::set_default_dtype(c10::scalarTypeToTypeMeta(old_dtype_));
  }
};

TEST_P(BinaryIntToFloatPromotion, div) {
  const auto& testParams = GetParam();
  const auto dtype0 = std::get<0>(testParams);
  const auto dtype1 = std::get<1>(testParams);
  const auto default_dtype = std::get<2>(testParams);
  GenerateInputs(2, {dtype0, dtype1});

  torch::set_default_dtype(c10::scalarTypeToTypeMeta(default_dtype));

  auto exp = torch::div(GetCpuInput(0), GetCpuInput(1));
  auto res = torch::div(GetHpuInput(0), GetHpuInput(1));
  Compare(exp, res, 0.01, 0.01);
}

INSTANTIATE_TEST_SUITE_P(
    TypePromotion,
    BinaryIntToFloatPromotion,
    ::testing::Combine(
        ::testing::ValuesIn(DtypesList),
        ::testing::ValuesIn(DtypesList),
        ::testing::ValuesIn(DefaultDtypesList)),
    PrintParam3());

/////////////////////////////////////////////////////////////////////////////////////

class UnaryIntToFloatPromotion
    : public HpuOpTestUtil,
      public testing::WithParamInterface<std::tuple<c10::ScalarType>> {};

TEST_P(UnaryIntToFloatPromotion, reciprocal) {
  const auto& testParams = GetParam();
  const auto dtype = std::get<0>(testParams);
  GenerateInputs(1, dtype);

  auto exp = torch::reciprocal(GetCpuInput(0));
  auto res = torch::reciprocal(GetHpuInput(0));
  Compare(exp, res);
}

INSTANTIATE_TEST_SUITE_P(
    TypePromotion,
    UnaryIntToFloatPromotion,
    ::testing::Combine(::testing::ValuesIn(DtypesList)),
    PrintParam1());

/////////////////////////////////////////////////////////////////////////////////////

class TensorListPromotion
    : public HpuOpTestUtil,
      public testing::WithParamInterface<
          std::tuple<c10::ScalarType, c10::ScalarType, c10::ScalarType>> {};

TEST_P(TensorListPromotion, cat) {
  const auto& testParams = GetParam();
  const auto dtype0 = std::get<0>(testParams);
  const auto dtype1 = std::get<1>(testParams);
  const auto dtype2 = std::get<2>(testParams);
  GenerateInputs(3, {dtype0, dtype1, dtype2});

  auto exp = torch::cat({GetCpuInput(0), GetCpuInput(1), GetCpuInput(2)});
  auto res = torch::cat({GetHpuInput(0), GetHpuInput(1), GetHpuInput(2)});
  Compare(exp, res);
}

INSTANTIATE_TEST_SUITE_P(
    TypePromotion,
    TensorListPromotion,
    ::testing::Combine(
        ::testing::ValuesIn(DtypesList),
        ::testing::ValuesIn(DtypesList),
        ::testing::ValuesIn(DtypesList)),
    PrintParam3());
