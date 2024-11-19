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