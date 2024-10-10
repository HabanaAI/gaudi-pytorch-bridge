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

TEST_F(HpuOpTest, cosine_embedding_loss) {
  GenerateInputs(3, {{5, 6}, {5, 6}, {5}});
  float margin = GenerateScalar<float>(0, 0.5);
  int reduction = GenerateScalar<int>(0, 2);
  auto expected = torch::cosine_embedding_loss(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), margin, reduction);
  auto result = torch::cosine_embedding_loss(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), margin, reduction);
  Compare(expected, result);
}

TEST_F(HpuOpTest, cosine_similarity) {
  GenerateInputs(2);
  float eps = GenerateScalar<float>(1e-10, 1e-7);
  int dim = -1;
  auto expected =
      torch::cosine_similarity(GetCpuInput(0), GetCpuInput(1), dim, eps);
  auto result =
      torch::cosine_similarity(GetHpuInput(0), GetHpuInput(1), dim, eps);
  Compare(expected, result);
}