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
#include "hint_tolerance_values.h"
#include <torch/torch.h>
#include <sstream>

std::string HintToleranceValues(
    const at::Tensor& result_hpu,
    const at::Tensor& result_cpu,
    float atol,
    float rtol) {
  std::stringstream ss;
  auto abs_diff = (result_hpu - result_cpu).abs();
  auto abs_cpu = result_cpu.abs();
  ss << "Maximum abs diff = "
     << (result_hpu - result_cpu).abs().max().to(torch::kFloat32).item<float>()
     << ", Required Atol = Rtol >= "
     << (abs_diff / (1 + abs_cpu)).max().to(torch::kFloat32).item<float>()
     << " OR Atol = " << atol << " and Rtol >= "
     << ((abs_diff - atol) / abs_cpu).max().to(torch::kFloat32).item<float>()
     << " OR Atol >= "
     << (abs_diff - rtol * abs_cpu).max().to(torch::kFloat32).item<float>()
     << " and Rtol = " << rtol;
  return ss.str();
}