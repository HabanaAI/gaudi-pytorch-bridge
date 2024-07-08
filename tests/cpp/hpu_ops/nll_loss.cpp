/******************************************************************************
 * Copyright (C) 2021-2024 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "../utils/device_type_util.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, nll_loss_fwd) {
  GenerateIntInputs(1, {{3}}, 0, 9);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);

  GenerateInputs(2, {{3, 10}, {10}});
  int reduction = torch::Reduction::Sum;

  auto expected = torch::nll_loss_forward(
      GetCpuInput(0),
      target,
      GetCpuInput(1) /*weight*/,
      reduction,
      2 /* ignore_index */);
  auto result = torch::nll_loss_forward(
      GetHpuInput(0),
      htarget,
      GetHpuInput(1) /*weight*/,
      reduction,
      2 /* ignore_index */);
  Compare(std::get<0>(expected), std::get<0>(result));
}
TEST_F(HpuOpTest, nll_loss_bwd) {
  GenerateIntInputs(1, {{3}}, 0, 9);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);

  GenerateInputs(4, {{3}, {3, 10}, {10}, {1}}, {torch::kBFloat16});
  int reduction = torch::Reduction::None;

  auto expected = torch::nll_loss_backward(
      GetCpuInput(0),
      GetCpuInput(1),
      target,
      GetCpuInput(2), /*weight*/
      reduction,
      2, /* ignore_index */
      GetCpuInput(3));
  auto result = torch::nll_loss_backward(
      GetHpuInput(0),
      GetHpuInput(1),
      htarget,
      GetHpuInput(2), /*weight*/
      reduction,
      2, /* ignore_index */
      GetHpuInput(3));
  Compare(expected, result);
}

TEST_F(HpuOpTest, nll_loss2d_fwd) {
  GenerateIntInputs(1, {{16, 8, 2}}, 0, 23);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);

  GenerateInputs(2, {{16, 24, 8, 2}, {24}}, {torch::kBFloat16});
  int reduction = torch::Reduction::None;

  auto expected = torch::nll_loss2d_forward(
      GetCpuInput(0),
      target,
      GetCpuInput(1) /*weight*/,
      reduction,
      2 /* ignore_index */);
  auto result = torch::nll_loss2d_forward(
      GetHpuInput(0),
      htarget,
      GetHpuInput(1) /*weight*/,
      reduction,
      2 /* ignore_index */);
  Compare(std::get<0>(expected), std::get<0>(result));
}

TEST_F(HpuOpTest, nll_loss2d_bwd) {
  GenerateIntInputs(1, {{5, 64, 64}}, 0, 2);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);

  GenerateInputs(4, {{1}, {5, 3, 64, 64}, {3}, {1}});
  int reduction = torch::Reduction::Sum;

  auto expected = torch::nll_loss2d_backward(
      GetCpuInput(0),
      GetCpuInput(1),
      target,
      GetCpuInput(2), /*weight*/
      reduction,
      3, /* ignore_index */
      GetCpuInput(3));
  auto result = torch::nll_loss2d_backward(
      GetHpuInput(0),
      GetHpuInput(1),
      htarget,
      GetHpuInput(2), /*weight*/
      reduction,
      3, /* ignore_index */
      GetHpuInput(3));
  Compare(expected, result);
}

TEST_F(HpuOpTest, nll_loss_fwd_out) {
  GenerateIntInputs(1, {{4}}, 0, 3);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);

  GenerateInputs(1, {{4, 5}});
  int reduction = torch::Reduction::None;
  torch::ScalarType dtype = torch::kFloat;

  auto out = torch::empty(0, dtype);
  auto hout = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto total_weight = torch::empty(0, dtype);
  auto htotal_weight =
      torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::nll_loss_forward_outf(
      GetCpuInput(0),
      target,
      {} /* weight */,
      reduction,
      -100 /* ignore_index */,
      out,
      total_weight);
  torch::nll_loss_forward_outf(
      GetHpuInput(0),
      htarget,
      {} /* weight */,
      reduction,
      -100 /* ignore_index */,
      hout,
      htotal_weight);
  Compare(out, hout, 0, 0);
}

TEST_F(HpuOpTest, nll_loss2d_fwd_out_bf16) {
  GenerateIntInputs(1, {{4, 2, 4}}, 0, 5);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);
  torch::ScalarType dtype = torch::kBFloat16;

  GenerateInputs(1, {{4, 6, 2, 4}}, {dtype});
  int reduction = torch::Reduction::Mean;

  auto out = torch::empty(0, dtype);
  auto hout = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto total_weight = torch::empty(0, dtype);
  auto htotal_weight =
      torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::nll_loss2d_forward_outf(
      GetCpuInput(0),
      target,
      {} /* weight */,
      reduction,
      1 /* ignore_index */,
      out,
      total_weight);
  torch::nll_loss2d_forward_outf(
      GetHpuInput(0),
      htarget,
      {} /* weight */,
      reduction,
      1 /* ignore_index */,
      hout,
      htotal_weight);
  Compare(out, hout);
}

TEST_F(HpuOpTest, nll_loss_bwd_out_bf16) {
  GenerateIntInputs(1, {{4}}, 0, 3);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(2, {{1}, {4, 5}}, {dtype, dtype});

  auto sum_weights = torch::sum(target, dtype);
  auto hsum_weights = sum_weights.to(torch::kHPU);
  int reduction = torch::Reduction::Mean;

  auto out = torch::empty(0, dtype);
  auto hout = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::nll_loss_backward_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      target,
      {} /* weight */,
      reduction,
      -10 /* ignore_index */,
      sum_weights,
      out);
  torch::nll_loss_backward_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      htarget,
      {} /* weight */,
      reduction,
      -10 /* ignore_index */,
      hsum_weights,
      hout);
  Compare(out, hout, 0, 0);
}

TEST_F(HpuOpTest, nll_loss2d_bwd_out) {
  GenerateIntInputs(1, {{2, 3, 5}}, 0, 4);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);

  GenerateInputs(2, {{1}, {2, 4, 3, 5}});
  int reduction = torch::Reduction::Sum;
  torch::ScalarType dtype = torch::kFloat;

  auto total_weight = torch::sum(target).to(dtype);
  auto htotal_weight = total_weight.to("hpu");
  auto out = torch::empty(0, dtype);
  auto hout = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::nll_loss2d_backward_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      target,
      {} /* weight */,
      reduction,
      10 /* ignore_index */,
      total_weight,
      out);
  torch::nll_loss2d_backward_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      htarget,
      {} /* weight */,
      reduction,
      10 /* ignore_index */,
      htotal_weight,
      hout);
  Compare(out, hout, 0, 0);
}

/*
// Note: Issue raise: https://jira.habana-labs.com/browse/SW-81413
// NllLoss HPU output doesn't match with CPU output
// when weight tensor is not None and reduction mode is Mean

TEST_F(HpuOpTest, nll_loss_with_mean) {
  GenerateIntInputs(1, {{3}}, 0, 9);
  auto target = GetCpuInput(0).to(torch::kLong);
  auto htarget = GetHpuInput(0).to(torch::kLong);

  GenerateInputs(2, {{3, 10}, {10}});
  int reduction = torch::Reduction::Mean;

  auto expected = torch::nll_loss_forward(
      GetCpuInput(0),
      target,
      GetCpuInput(1), // weight
      reduction,
      2);
  auto result = torch::nll_loss_forward(
      GetHpuInput(0),
      htarget,
      GetHpuInput(1), // weight
      reduction,
      2);
  Compare(std::get<0>(expected), std::get<0>(result));
}*/