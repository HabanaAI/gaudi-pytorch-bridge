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

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, LogSoftMaxOutTest1D) {
  std::vector<int64_t> size = {1024};
  GenerateInputs(2, {size, size});

  torch::_log_softmax_outf(GetCpuInput(0), /*dim*/ 0, false, GetCpuInput(1));
  torch::_log_softmax_outf(GetHpuInput(0), /*dim*/ 0, false, GetHpuInput(1));

  Compare(GetCpuInput(1), GetHpuInput(1));
}

/**
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-67907
 */
TEST_F(HpuOpTest, LogSoftMaxOutTest2D) {
  std::vector<int64_t> size = {128, 256};
  GenerateInputs(2, {size, size}, {torch::kBFloat16});

  torch::_log_softmax_outf(GetCpuInput(0), /*dim*/ 1, false, GetCpuInput(1));
  torch::_log_softmax_outf(GetHpuInput(0), /*dim*/ 1, false, GetHpuInput(1));

  Compare(GetCpuInput(1), GetHpuInput(1), 1.3e-2, 1.3e-2);
}

TEST_F(HpuOpTest, LogSoftMaxOutTest3D) {
  std::vector<int64_t> size = {128, 64, 64};
  GenerateInputs(2, {size, size});

  torch::_log_softmax_outf(GetCpuInput(0), /*dim*/ -1, false, GetCpuInput(1));
  torch::_log_softmax_outf(GetHpuInput(0), /*dim*/ -1, false, GetHpuInput(1));

  Compare(GetCpuInput(1), GetHpuInput(1));
}

/**
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-67907
 */
TEST_F(HpuOpTest, LogSoftMaxOutTest4D) {
  std::vector<int64_t> size = {8, 3, 64, 64};
  GenerateInputs(2, {size, size}, {torch::kBFloat16});

  torch::_log_softmax_outf(GetCpuInput(0), /*dim*/ 2, false, GetCpuInput(1));
  torch::_log_softmax_outf(GetHpuInput(0), /*dim*/ 2, false, GetHpuInput(1));

  Compare(GetCpuInput(1), GetHpuInput(1), 1.3e-2, 1.3e-2);
}

TEST_F(HpuOpTest, LogSoftMaxOutTest5D) {
  std::vector<int64_t> size = {8, 3, 64, 64, 64};
  GenerateInputs(2, {size, size});

  torch::_log_softmax_outf(GetCpuInput(0), /*dim*/ -3, false, GetCpuInput(1));
  torch::_log_softmax_outf(GetHpuInput(0), /*dim*/ -3, false, GetHpuInput(1));

  Compare(GetCpuInput(1), GetHpuInput(1));
}

TEST_F(HpuOpTest, LogSoftMaxBwdOutTest1D) {
  std::vector<int64_t> size = {1024};
  GenerateInputs(4, {size, size, size, size});

  torch::_log_softmax_backward_data_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      /*dim*/ 0,
      GetCpuInput(2).scalar_type(),
      GetCpuInput(3));
  torch::_log_softmax_backward_data_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      /*dim*/ 0,
      GetHpuInput(2).scalar_type(),
      GetHpuInput(3));

  Compare(GetCpuInput(3), GetHpuInput(3));
}

/**
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-67907
 */
TEST_F(HpuOpTest, LogSoftMaxBwdOutTest2D) {
  std::vector<int64_t> size = {128, 256};
  GenerateInputs(4, {size, size, size, size}, {torch::kBFloat16});

  torch::_log_softmax_backward_data_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      /*dim*/ -1,
      GetCpuInput(2).scalar_type(),
      GetCpuInput(3));
  torch::_log_softmax_backward_data_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      /*dim*/ -1,
      GetHpuInput(2).scalar_type(),
      GetHpuInput(3));

  Compare(GetCpuInput(3), GetHpuInput(3), 1.4e-2, 1.4e-2);
}

TEST_F(HpuOpTest, LogSoftMaxBwdOutTest3D) {
  std::vector<int64_t> size = {128, 64, 64};
  GenerateInputs(4, {size, size, size, size});

  torch::_log_softmax_backward_data_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      /*dim*/ 2,
      GetCpuInput(2).scalar_type(),
      GetCpuInput(3));
  torch::_log_softmax_backward_data_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      /*dim*/ 2,
      GetHpuInput(2).scalar_type(),
      GetHpuInput(3));

  Compare(GetCpuInput(3), GetHpuInput(3));
}

/**
 * Below test will fail for BFloat16 for default tolerance
 * Issue raised: https://jira.habana-labs.com/browse/SW-67907
 */
TEST_F(HpuOpTest, LogSoftMaxBwdOutTest4D) {
  std::vector<int64_t> size = {8, 3, 64, 64};
  GenerateInputs(4, {size, size, size, size}, {torch::kBFloat16});

  torch::_log_softmax_backward_data_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      /*dim*/ -2,
      GetCpuInput(2).scalar_type(),
      GetCpuInput(3));
  torch::_log_softmax_backward_data_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      /*dim*/ -2,
      GetHpuInput(2).scalar_type(),
      GetHpuInput(3));

  Compare(GetCpuInput(3), GetHpuInput(3), 2.8e-2, 2.8e-2);
}

TEST_F(HpuOpTest, LogSoftMaxBwdOutTest5D) {
  std::vector<int64_t> size = {8, 3, 64, 64, 64};
  GenerateInputs(4, {size, size, size, size});

  torch::_log_softmax_backward_data_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      /*dim*/ -3,
      GetCpuInput(2).scalar_type(),
      GetCpuInput(3));
  torch::_log_softmax_backward_data_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      /*dim*/ -3,
      GetHpuInput(2).scalar_type(),
      GetHpuInput(3));

  Compare(GetCpuInput(3), GetHpuInput(3));
}
