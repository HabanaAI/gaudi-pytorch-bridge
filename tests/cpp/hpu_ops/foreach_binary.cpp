/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, foreach_add_scalar) {
  static constexpr int n = 4;
  std::vector<at::Tensor> cpu_in;
  std::vector<at::Tensor> hpu_in;
  std::vector<std::vector<long>> sizes = {
      {4, 2, 3}, {4, 0, 5}, {128}, {64, 1}, {2, 3, 4, 5}};
  std::vector<torch::ScalarType> dtypes = {
      at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16};

  for (int i = 0; i < n; ++i) {
    GenerateInputs(1, {sizes[i]}, dtypes[i]);
    cpu_in.push_back(GetCpuInput(0));
    hpu_in.push_back(GetHpuInput(0));
  }

  auto exp = _foreach_add(cpu_in, 1.31234579);
  auto res = _foreach_add(hpu_in, 1.31234579);

  for (int i = 0; i < n; ++i) {
    Compare(exp[i], res[i]);
  }
}

TEST_F(HpuOpTest, foreach_add_list) {
  static constexpr int n = 5;
  std::vector<at::Tensor> cpu_in1, cpu_in2;
  std::vector<at::Tensor> hpu_in1, hpu_in2;
  std::vector<std::vector<long>> sizes1 = {
      {4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}};
  std::vector<std::vector<long>> sizes2 = {
      {4, 1, 3}, {4, 1, 5}, {7}, {0}, {2, 3, 4, 5}};
  std::vector<torch::ScalarType> dtypes1 = {
      at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16};
  std::vector<torch::ScalarType> dtypes2 = {
      at::kByte, at::kDouble, at::kLong, at::kShort, at::kInt};

  for (int i = 0; i < n; ++i) {
    GenerateInputs(2, {sizes1[i], sizes2[i]}, {dtypes1[i], dtypes2[i]});

    cpu_in1.push_back(GetCpuInput(0));
    cpu_in2.push_back(GetCpuInput(1));

    hpu_in1.push_back(GetHpuInput(0));
    hpu_in2.push_back(GetHpuInput(1));
  }

  auto exp = _foreach_add(cpu_in1, cpu_in2, 3);
  auto res = _foreach_add(hpu_in1, hpu_in2, 3);

  for (int i = 0; i < n; ++i) {
    Compare(exp[i], res[i]);
  }
}

TEST_F(HpuOpTest, foreach_add_scalarlist) {
  static constexpr int n = 5;
  std::vector<at::Tensor> cpu_in;
  std::vector<at::Tensor> hpu_in;
  std::vector<std::vector<long>> sizes = {
      {4, 2, 3}, {5}, {7}, {64, 0}, {2, 1, 4, 1}};
  std::vector<torch::ScalarType> dtypes = {
      at::kInt, at::kFloat, at::kByte, at::kLong, at::kBFloat16};
  std::vector<torch::Scalar> s = {7, 3.141, 2., -100, -0.001};

  for (int i = 0; i < n; ++i) {
    GenerateInputs(1, {sizes[i]}, {dtypes[i]});

    cpu_in.push_back(GetCpuInput(0));
    hpu_in.push_back(GetHpuInput(0));
  }

  auto exp = _foreach_add(cpu_in, s);
  auto res = _foreach_add(hpu_in, s);

  for (int i = 0; i < n; ++i) {
    Compare(exp[i], res[i]);
  }
}
