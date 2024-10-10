/******************************************************************************
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

typedef const std::function<
    std::vector<at::Tensor>(at::TensorList, at::TensorList)>
    FunctionTwoLists;
typedef const std::function<std::vector<at::Tensor>(at::TensorList)>
    FunctionOneList;
typedef const std::function<void(at::TensorList)> FunctionOneListInplace;
typedef const std::function<void(at::TensorList, at::TensorList)>
    FunctionTwoListsInplace;

std::vector<at::Scalar> scalars = {7, 3.141, 2., -100, -0.001};

class HpuOpTest : public HpuOpTestUtil {
 private:
  void compareResults(
      std::vector<at::Tensor> expected,
      std::vector<at::Tensor> results) {
    EXPECT_TRUE(expected.size() == results.size());
    for (int i = 0; i < expected.size(); ++i) {
      Compare(expected[i], results[i]);
    }
  }

 public:
  void TestForeachBinary(
      std::vector<std::vector<long>> sizes,
      std::vector<at::ScalarType> dtypes,
      FunctionOneList& fn) {
    std::vector<at::Tensor> cpu_in;
    std::vector<at::Tensor> hpu_in;
    for (int i = 0; i < sizes.size(); ++i) {
      GenerateInputs(1, {sizes[i]}, dtypes[i]);
      cpu_in.push_back(GetCpuInput(0));
      hpu_in.push_back(GetHpuInput(0));
    }

    auto exp = fn(cpu_in);
    auto res = fn(hpu_in);

    compareResults(exp, res);
  }

  void TestForeachBinaryList(
      std::vector<std::vector<long>> sizes,
      std::vector<std::vector<long>> otherSizes,
      std::vector<at::ScalarType> dtypes,
      std::vector<at::ScalarType> otherDtypes,
      FunctionTwoLists& fn) {
    std::vector<at::Tensor> cpu_in1, cpu_in2;
    std::vector<at::Tensor> hpu_in1, hpu_in2;
    for (int i = 0; i < sizes.size(); ++i) {
      GenerateInputs(2, {sizes[i], otherSizes[i]}, {dtypes[i], otherDtypes[i]});

      cpu_in1.push_back(GetCpuInput(0));
      cpu_in2.push_back(GetCpuInput(1));

      hpu_in1.push_back(GetHpuInput(0));
      hpu_in2.push_back(GetHpuInput(1));
    }

    auto exp = fn(cpu_in1, cpu_in2);
    auto res = fn(hpu_in1, hpu_in2);

    compareResults(exp, res);
  }

  void TestForeachBinaryInplace(
      std::vector<std::vector<long>> sizes,
      std::vector<at::ScalarType> dtypes,
      FunctionOneListInplace& fn) {
    std::vector<at::Tensor> cpu_in;
    std::vector<at::Tensor> hpu_in;
    for (int i = 0; i < sizes.size(); ++i) {
      GenerateInputs(1, {sizes[i]}, dtypes[i]);
      cpu_in.push_back(GetCpuInput(0));
      hpu_in.push_back(GetHpuInput(0));
    }

    fn(cpu_in);
    fn(hpu_in);

    compareResults(cpu_in, hpu_in);
  }

  void TestForeachBinaryListInplace(
      std::vector<std::vector<long>> sizes,
      std::vector<std::vector<long>> otherSizes,
      std::vector<at::ScalarType> dtypes,
      std::vector<at::ScalarType> otherDtypes,
      const std::function<void(at::TensorList, at::TensorList)>& fn) {
    std::vector<at::Tensor> cpu_in1, cpu_in2;
    std::vector<at::Tensor> hpu_in1, hpu_in2;
    for (int i = 0; i < sizes.size(); ++i) {
      GenerateInputs(2, {sizes[i], otherSizes[i]}, {dtypes[i], otherDtypes[i]});

      cpu_in1.push_back(GetCpuInput(0));
      cpu_in2.push_back(GetCpuInput(1));

      hpu_in1.push_back(GetHpuInput(0));
      hpu_in2.push_back(GetHpuInput(1));
    }

    fn(cpu_in1, cpu_in2);
    fn(hpu_in1, hpu_in2);

    compareResults(cpu_in1, hpu_in1);
  }
};

TEST_F(HpuOpTest, foreachAdd) {
  FunctionOneList foreach_add_scalar = std::bind(
      static_cast<std::vector<at::Tensor> (*)(
          at::TensorList, const at::Scalar&)>(at::_foreach_add),
      std::placeholders::_1,
      1.31234579);
  TestForeachBinary(
      {{4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}},
      {at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16},
      foreach_add_scalar);

  FunctionOneList foreach_add_scalars = std::bind(
      static_cast<std::vector<at::Tensor> (*)(
          at::TensorList, at::ArrayRef<at::Scalar>)>(at::_foreach_add),
      std::placeholders::_1,
      scalars);
  TestForeachBinary(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}},
      {at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16},
      foreach_add_scalars);

  FunctionTwoLists foreach_add_list = std::bind(
      static_cast<std::vector<at::Tensor> (*)(
          at::TensorList, at::TensorList, const at::Scalar&)>(at::_foreach_add),
      std::placeholders::_1,
      std::placeholders::_2,
      3);
  TestForeachBinaryList(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}},
      {{4, 1, 3}, {4, 1, 5}, {7}, {0}, {2, 3, 4, 5}},
      {at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16},
      {at::kByte, at::kDouble, at::kLong, at::kShort, at::kInt},
      foreach_add_list);
}

TEST_F(HpuOpTest, foreachAddInplace) {
  FunctionOneListInplace foreach_add_inplace_scalar_floats = std::bind(
      static_cast<void (*)(at::TensorList, const at::Scalar&)>(
          at::_foreach_add_),
      std::placeholders::_1,
      2.6431);
  TestForeachBinaryInplace(
      {{4, 3, 5}, {2, 3, 4, 5}},
      {at::kFloat, at::kBFloat16},
      foreach_add_inplace_scalar_floats);

  FunctionOneListInplace foreach_add_inplace_scalar_ints = std::bind(
      static_cast<void (*)(at::TensorList, const at::Scalar&)>(
          at::_foreach_add_),
      std::placeholders::_1,
      2);
  TestForeachBinaryInplace(
      {{4, 3, 5}, {2, 3, 4, 5}},
      {at::kInt, at::kLong},
      foreach_add_inplace_scalar_ints);

  FunctionOneListInplace foreach_add_inplace_scalars = std::bind(
      static_cast<void (*)(at::TensorList, at::ArrayRef<at::Scalar>)>(
          at::_foreach_add_),
      std::placeholders::_1,
      scalars);
  TestForeachBinaryInplace(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}},
      {at::kInt, at::kFloat, at::kFloat, at::kLong, at::kBFloat16},
      foreach_add_inplace_scalars);

  FunctionTwoListsInplace foreach_add_inplace_list = std::bind(
      static_cast<void (*)(at::TensorList, at::TensorList, const at::Scalar&)>(
          at::_foreach_add_),
      std::placeholders::_1,
      std::placeholders::_2,
      3);
  TestForeachBinaryListInplace(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 3, 4, 5}},
      {{4, 1, 1}, {5}, {7}, {64, 0}, {2, 1, 4, 5}},
      {at::kFloat, at::kFloat, at::kFloat, at::kShort, at::kInt},
      {at::kInt, at::kBFloat16, at::kFloat, at::kShort, at::kShort},
      foreach_add_inplace_list);
}

TEST_F(HpuOpTest, foreachMul) {
  FunctionOneList foreach_mul_scalar = std::bind(
      static_cast<std::vector<at::Tensor> (*)(
          at::TensorList, const at::Scalar&)>(at::_foreach_mul),
      std::placeholders::_1,
      2.6431);
  TestForeachBinary(
      {{4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}},
      {at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16},
      foreach_mul_scalar);

  FunctionOneList foreach_mul_scalars = std::bind(
      static_cast<std::vector<at::Tensor> (*)(
          at::TensorList, at::ArrayRef<at::Scalar>)>(at::_foreach_mul),
      std::placeholders::_1,
      scalars);
  TestForeachBinary(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}},
      {at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16},
      foreach_mul_scalars);

  TestForeachBinaryList(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}},
      {{4, 1, 3}, {4, 1, 5}, {7}, {0}, {2, 3, 4, 5}},
      {at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16},
      {at::kByte, at::kDouble, at::kLong, at::kShort, at::kInt},
      static_cast<std::vector<at::Tensor> (*)(at::TensorList, at::TensorList)>(
          at::_foreach_mul));
}

TEST_F(HpuOpTest, foreachMulInplace) {
  FunctionOneListInplace foreach_mul_inplace_scalar_floats = std::bind(
      static_cast<void (*)(at::TensorList, const at::Scalar&)>(
          at::_foreach_mul_),
      std::placeholders::_1,
      2.6431);
  TestForeachBinaryInplace(
      {{4, 3, 5}, {2, 3, 4, 5}},
      {at::kFloat, at::kBFloat16},
      foreach_mul_inplace_scalar_floats);

  FunctionOneListInplace foreach_mul_inplace_scalar_ints = std::bind(
      static_cast<void (*)(at::TensorList, const at::Scalar&)>(
          at::_foreach_mul_),
      std::placeholders::_1,
      2);
  TestForeachBinaryInplace(
      {{4, 3, 5}, {2, 3, 4, 5}},
      {at::kInt, at::kLong},
      foreach_mul_inplace_scalar_ints);

  FunctionOneListInplace foreach_mul_inplace_scalars = std::bind(
      static_cast<void (*)(at::TensorList, at::ArrayRef<at::Scalar>)>(
          at::_foreach_mul_),
      std::placeholders::_1,
      scalars);
  TestForeachBinaryInplace(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}},
      {at::kInt, at::kFloat, at::kFloat, at::kLong, at::kBFloat16},
      foreach_mul_inplace_scalars);

  TestForeachBinaryListInplace(
      {{4, 2, 3}, {5}, {7}, {64, 0}, {2, 3, 4, 5}},
      {{4, 1, 1}, {5}, {7}, {64, 0}, {2, 1, 4, 5}},
      {at::kFloat, at::kFloat, at::kFloat, at::kShort, at::kInt},
      {at::kInt, at::kBFloat16, at::kFloat, at::kShort, at::kShort},
      static_cast<void (*)(at::TensorList, at::TensorList)>(at::_foreach_mul_));
}
