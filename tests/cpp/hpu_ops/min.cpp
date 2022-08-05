/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"
using namespace std;

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, min_keepdim_true) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), 2 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 2 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_keepdim_false) {
  GenerateInputs(1, {{2, 3, 4, 5}}, {torch::kBFloat16});
  auto expected = torch::min(GetCpuInput(0), 2 /*dim*/, false /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 2 /*dim*/, false /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_neg_dim_keepdim_true) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), -2 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), -2 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_neg_dim_keepdim_false) {
  GenerateInputs(1, {{2, 3, 4, 5}}, {torch::kBFloat16});
  auto expected = torch::min(GetCpuInput(0), -2 /*dim*/, false /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), -2 /*dim*/, false /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_dim_0_keepdim_true) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), 0 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 0 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_dim_0_keepdim_false) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), 0 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 0 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_out_keepdim_true) {
  GenerateInputs(1, {{2, 3, 4, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto min_val = torch::empty(0, dtype);
  auto hmin_val = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto min_indice = torch::empty(0, torch::kLong);
  auto hmin_indice = min_indice.to("hpu");

  torch::min_outf(
      GetCpuInput(0), 2 /*dim*/, true /*keepdim*/, min_val, min_indice);
  torch::min_outf(
      GetHpuInput(0), 2 /*dim*/, true /*keepdim*/, hmin_val, hmin_indice);
  Compare(min_val, hmin_val);
  Compare(min_indice, hmin_indice);
}

TEST_F(HpuOpTest, min_out_keepdim_false) {
  GenerateInputs(1, {{2, 3, 4, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto min_val = torch::empty(0, dtype);
  auto hmin_val = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto min_indice = torch::empty(0, torch::kLong);
  auto hmin_indice = min_indice.to("hpu");

  torch::min_outf(
      GetCpuInput(0), 2 /*dim*/, false /*keepdim*/, min_val, min_indice);
  torch::min_outf(
      GetHpuInput(0), 2 /*dim*/, false /*keepdim */, hmin_val, hmin_indice);
  Compare(min_val, hmin_val);
  Compare(min_indice, hmin_indice);
}

TEST_F(HpuOpTest, min_out_dim_0_keepdim_true) {
  GenerateInputs(1, {{2, 3, 4, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto min_val = torch::empty(0, dtype);
  auto hmin_val = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto min_indice = torch::empty(0, torch::kLong);
  auto hmin_indice = min_indice.to("hpu");

  torch::min_outf(
      GetCpuInput(0), 0 /*dim*/, true /*keepdim*/, min_val, min_indice);
  torch::min_outf(
      GetHpuInput(0), 0 /*dim*/, true /*keepdim*/, hmin_val, hmin_indice);
  Compare(min_val, hmin_val);
  Compare(min_indice, hmin_indice);
}

TEST_F(HpuOpTest, min_out_dim_0_keepdim_false) {
  GenerateInputs(1, {{2, 3, 4, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto min_val = torch::empty(0, dtype);
  auto hmin_val = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto min_indice = torch::empty(0, torch::kLong);
  auto hmin_indice = min_indice.to("hpu");

  torch::min_outf(
      GetCpuInput(0), 0 /*dim*/, false /*keepdim*/, min_val, min_indice);
  torch::min_outf(
      GetHpuInput(0), 0 /*dim*/, false /*keepdim */, hmin_val, hmin_indice);
  Compare(min_val, hmin_val);
  Compare(min_indice, hmin_indice);
}

TEST_F(HpuOpTest, min_out_neg_keepdim_true) {
  GenerateInputs(1, {{2, 3, 4, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto min_val = torch::empty(0, dtype);
  auto hmin_val = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto min_indice = torch::empty(0, torch::kLong);
  auto hmin_indice = min_indice.to("hpu");

  torch::min_outf(
      GetCpuInput(0), -3 /*dim*/, true /*keepdim*/, min_val, min_indice);
  torch::min_outf(
      GetHpuInput(0), -3 /*dim*/, true /*keepdim*/, hmin_val, hmin_indice);
  Compare(min_val, hmin_val);
  Compare(min_indice, hmin_indice);
}

TEST_F(HpuOpTest, min_out_neg_keepdim_false) {
  GenerateInputs(1, {{2, 3, 4, 5}});

  torch::ScalarType dtype = torch::kFloat;
  auto min_val = torch::empty(0, dtype);
  auto hmin_val = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto min_indice = torch::empty(0, torch::kLong);
  auto hmin_indice = min_indice.to("hpu");

  torch::min_outf(
      GetCpuInput(0), -3 /*dim*/, false /*keepdim*/, min_val, min_indice);
  torch::min_outf(
      GetHpuInput(0), -3 /*dim*/, false /*keepdim */, hmin_val, hmin_indice);
  Compare(min_val, hmin_val);
  Compare(min_indice, hmin_indice);
}

TEST_F(HpuOpTest, min_out) {
  GenerateInputs(2, {{2, 2}, {2, 2}});

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = expected.to("hpu");

  torch::min_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::min_outf(GetHpuInput(0), GetHpuInput(1), result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, min_out_with_broadcast) {
  GenerateInputs(2, {{2, 1, 2}, {5, 2}});

  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = expected.to("hpu");

  torch::min_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::min_outf(GetHpuInput(0), GetHpuInput(1), result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, min_usual) {
  GenerateInputs(2, {{2, 2}, {2, 2}});
  auto expected = torch::min(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::min(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result);
}

TEST_F(HpuOpTest, min_usual_with_broadcast) {
  GenerateInputs(2, {{2, 1, 2}, {5, 2}});
  auto expected = torch::min(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::min(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result);
}

TEST_F(HpuOpTest, min_f32) {
  GenerateInputs(1, {{2, 2}});
  auto expected = torch::min(GetCpuInput(0));
  auto result = torch::min(GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, min_bf16) {
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {dtype}, {{2, 2}});
  auto expected = torch::min(GetCpuInput(0));
  auto result = torch::min(GetHpuInput(0));
  Compare(expected, result);
}
