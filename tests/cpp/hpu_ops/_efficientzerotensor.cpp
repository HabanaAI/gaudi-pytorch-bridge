/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, EfficientZeroTensor) {
  auto tensor = at::_efficientzerotensor(
      {{3, 2, 3}},
      torch::kFloat32,
      c10::nullopt,
      c10::Device(c10::DeviceType::HPU),
      c10::nullopt);
  auto expected_tensor = at::_efficientzerotensor(
      {{3, 2, 3}}, at::TensorOptions().dtype(torch::kFloat32));

  Compare(expected_tensor, tensor, 0, 0);
}
