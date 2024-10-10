/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
  static void clearPermuteInformation(const torch::Tensor& permutedTensor);

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