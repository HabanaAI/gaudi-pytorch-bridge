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

#pragma once
#include <torch/torch.h>
#include "backend/synapse_helpers/layout_utils.h"
namespace habana_lazy {

class PermuteTensors {
 public:
  PermuteTensors() = default;
  virtual ~PermuteTensors() = default;

  static void permuteWeight(torch::Tensor& weight);
  static void handlePermutedTensor(
      const torch::Tensor& permutedTensor,
      torch::Tensor& cpuTensor,
      bool non_blocking);

 private:
  static void setMemoryPermutation(
      const torch::Tensor& tensor,
      synapse_helpers::layouts::MemoryPermutation permutation);
  static synapse_helpers::layouts::MemoryPermutation getMemoryPermutation(
      const torch::Tensor& tensor);
  static void permuteWeightByDim(torch::Tensor& weight);
  static void permuteWeightToRSCKInMemory(torch::Tensor& weight);
  static void permuteWeightToQRSCKInMemory(torch::Tensor& weight);
  template <typename T>
  static void permuteWeightTensorDataToRSCK(const torch::Tensor& weight);
  template <typename T>
  static void restrideWeightTensorDataToQRSCK(const torch::Tensor& weight);
  static bool shouldPermuteWeight(const torch::Tensor& weight);
  static bool shouldPermutePreCastedWeight(const torch::Tensor& weight);
  static const torch::Tensor getPreCastedWeight(const torch::Tensor& weight);
  static unsigned m_permute_counter;
};

} // namespace habana_lazy