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

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, sum_4d_2d_keepdim) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  const std::vector<int64_t> dim{-2, 0};

  auto expected = torch::sum(GetCpuInput(0), dim, true /*keepdim*/);
  auto result = torch::sum(GetHpuInput(0), dim, true /*keepdim*/);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sum_3d_2d_keepdim_Int) {
  torch::ScalarType dtype = torch::kInt;
  GenerateInputs(1, {{5, 3, 6}}, dtype);
  const std::vector<int64_t> dim{-2, 0};

  auto expected = torch::sum(GetCpuInput(0), dim, true /*keepdim*/, dtype);
  auto result = torch::sum(GetHpuInput(0), dim, true /*keepdim*/, dtype);

  Compare(expected, result.to("hpu"));
}

TEST_F(HpuOpTest, sum_3d) {
  torch::ScalarType dtype = torch::kInt;
  GenerateInputs(1, {{5, 3, 6}}, dtype);
  //  const std::vector<int64_t> dim{-2, 0};

  auto expected = torch::sum(GetCpuInput(0), dtype);
  auto result = torch::sum(GetHpuInput(0), dtype);

  Compare(expected, result.to("hpu"));
}

TEST_F(HpuOpTest, sum_4d_2d_keepdim_Int) {
  torch::ScalarType dtype = torch::kInt;
  GenerateInputs(1, {{5, 3, 6, 4}}, dtype);
  const std::vector<int64_t> dim{-2, 0};

  auto expected = torch::sum(GetCpuInput(0), dim, true /*keepdim*/, dtype);
  auto result = torch::sum(GetHpuInput(0), dim, true /*keepdim*/, dtype);

  Compare(expected, result.to("hpu"));
}

TEST_F(HpuOpTest, sum_4d_2d_Int) {
  torch::ScalarType dtype = torch::kInt;
  GenerateInputs(1, {{5, 3, 6, 4}}, dtype);
  const std::vector<int64_t> dim{-2, 0};

  auto expected = torch::sum(GetCpuInput(0), dim, false /*keepdim*/, dtype);
  auto result = torch::sum(GetHpuInput(0), dim, false /*keepdim*/, dtype);

  Compare(expected, result.to("hpu"));
}

TEST_F(HpuOpTest, sum_4d_3d_reduce_dim) {
  GenerateInputs(1, {{3, 6, 5, 4}});
  const std::vector<int64_t> dim{3, 1, 0};

  auto expected = torch::sum(GetCpuInput(0), dim, false /*keepdim*/);
  auto result = torch::sum(GetHpuInput(0), dim, false /*keepdim*/);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sum) {
  GenerateInputs(1, {{3, 3, 4, 6}});

  auto expected = torch::sum(GetCpuInput(0));
  auto result = torch::sum(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, sum_bool) {
  GenerateInputs(1, {{3, 3, 4, 6}}, {{at::kBool}});

  auto expected = torch::sum(GetCpuInput(0));
  auto result = torch::sum(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, sum_UNET) {
  torch::ScalarType dtype = torch::kInt;
  GenerateInputs(1, {{5, 3, 6, 3}}, dtype);
  const std::vector<int64_t> dim{0, 2, 3};

  auto expected = torch::sum(GetCpuInput(0), dim, false /*keepdim*/, dtype);
  auto result = torch::sum(GetHpuInput(0), dim, false /*keepdim*/, dtype);

  Compare(expected, result.to("hpu"));
}

TEST_F(HpuOpTest, sum_4d_2d_keepdim_cmpt) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }

  GenerateInputs(1, {{2, 3, 4, 5}});
  const std::vector<int64_t> dim{-2, 0};

  auto expected = torch::sum(GetCpuInput(0), dim, true /*keepdim*/);
  auto result = torch::sum(GetHpuInput(0), dim, true /*keepdim*/);

  Compare(expected, result);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(HpuOpTest, sum_0d_keepdim_out) {
  GenerateInputs(1);
  auto scalar = c10::Scalar(GenerateScalar<float>());
  torch::ScalarType dtype = torch::kFloat;

  auto scalarCpu = torch::scalar_tensor(
      scalar, torch::TensorOptions().device("cpu").dtype(dtype));
  auto scalarHpu = scalarCpu.to("hpu");

  auto outCpu = torch::empty_like(scalarCpu);
  auto outHpu = torch::empty_like(scalarHpu);

  int64_t dim = 0;
  auto expected =
      torch::sum_outf(scalarCpu, dim, true /*keepdim*/, c10::nullopt, outCpu);
  auto result =
      torch::sum_outf(scalarHpu, dim, true /*keepdim*/, c10::nullopt, outHpu);

  Compare(expected, result);
}